/*
 * coach: the host side of the live training loop (docs/COACH.md). Watches the control block a `cluster-gate run` with
 * CX_COACH=N publishes in the host-memory part, decodes and scores the greedy samples, applies the plateau policy and
 * writes the learning-rate scale, save and stop knobs back. Nothing the coach writes is training data.
 *
 *   coach LIST.txt VOCAB.txt LOG.txt [--window W] [--min-gain BITS] [--rep4 R] [--halvings H] [--force-scale X] [--stop-at N]
 *
 * Policy (rule based): every W reports (default 10) the mean training loss of the window is compared with the previous
 * window; a gain below BITS bits/token (default 0.01) is a plateau; a plateau halves the scale; after H halvings
 * (default 3) the next plateau sets the stop flag. Three consecutive samples with rep4 above R (default 0.036, the p90
 * of human TinyStories windows) count as one plateau. --force-scale and --stop-at are test knobs.
 */
#define BPE_NO_MAIN
#include <stdarg.h>
#include "../tools/bpe.c"
#include "../tools/coherence.h"
#include "../lib/cx-proto.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#ifndef CX_LN2
#define CX_LN2 0.6931471805599453
#endif

static uint8_t *part;
#define LD64(o) __atomic_load_n((uint64_t *)(part + (o)), __ATOMIC_ACQUIRE)
#define LD32(o) __atomic_load_n((uint32_t *)(part + (o)), __ATOMIC_ACQUIRE)
#define ST64(o, x) __atomic_store_n((uint64_t *)(part + (o)), (x), __ATOMIC_RELEASE)
static double kdbl(uint64_t u) { double d; memcpy(&d, &u, 8); return d; }
static uint64_t kbits(double d) { uint64_t u; memcpy(&u, &d, 8); return u; }

static FILE *lg;
static void logf2(const char *fmt, ...)
{
	va_list ap;
	char line[8192];
	va_start(ap, fmt);
	vsnprintf(line, sizeof line, fmt, ap);
	va_end(ap);
	time_t now = time(NULL);
	struct tm tm;
	localtime_r(&now, &tm);
	char ts[32];
	strftime(ts, sizeof ts, "%H:%M:%S", &tm);
	printf("%s %s\n", ts, line);
	fflush(stdout);
	if (lg) { fprintf(lg, "%s %s\n", ts, line); fflush(lg); }
}

int main(int argc, char **argv)
{
	if (argc < 4) { fprintf(stderr, "usage: coach LIST.txt VOCAB.txt LOG.txt [--window W] [--min-gain BITS] [--rep4 R] [--halvings H] [--force-scale X] [--stop-at N]\n"); return 2; }
	int window = 10, max_halvings = 3; double min_gain = 0.01, rep4_band = 0.036, force_scale = 0; long stop_at = 0;
	for (int i = 4; i < argc; i++) {
		if (!strcmp(argv[i], "--window") && i + 1 < argc) window = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--min-gain") && i + 1 < argc) min_gain = atof(argv[++i]);
		else if (!strcmp(argv[i], "--rep4") && i + 1 < argc) rep4_band = atof(argv[++i]);
		else if (!strcmp(argv[i], "--halvings") && i + 1 < argc) max_halvings = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--force-scale") && i + 1 < argc) force_scale = atof(argv[++i]);
		else if (!strcmp(argv[i], "--stop-at") && i + 1 < argc) stop_at = atol(argv[++i]);
		else { fprintf(stderr, "coach: unknown option %s\n", argv[i]); return 2; }
	}
	if (coh_load(argv[1])) { perror(argv[1]); return 1; }
	if (load_vocab(argv[2])) return 1;
	for (int id = 0; id < 256 + nmerges; id++) expand(id);
	expand(EOT);
	lg = fopen(argv[3], "a");
	if (!lg) { perror(argv[3]); return 1; }
	const int fd = open("/dev/shm/phi-hostmem", O_RDWR);
	if (fd < 0) { perror("/dev/shm/phi-hostmem"); return 1; }
	part = mmap(NULL, CX_PART_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)CX_PART_MIB << 20);
	if (part == MAP_FAILED) { perror("mmap"); return 1; }
	logf2("coach: list %s, vocab %s, window %d reports, min gain %.3f bits/token, rep4 band %.3f, max halvings %d%s", argv[1], argv[2], window, min_gain, rep4_band, max_halvings, force_scale > 0 ? " (forced scale)" : "");

	int live = 0, halvings = 0, rep_streak = 0, forced = 0;
	uint64_t last_chunk = 0, last_seq = 0;
	double scale = 1.0, win_sum = 0, prev_mean = 0; int win_n = 0, have_prev = 0;
	for (;;) {
		usleep(200000);
		const int magic = LD32(CX_K_MAGIC) == CX_K_MAGIC_VAL;
		if (!magic) { if (live) { logf2("coach: run ended (magic cleared); %d halvings, final scale %.4f", halvings, scale); break; } continue; }
		if (!live) { live = 1; last_chunk = LD64(CX_K_CHUNK); last_seq = LD64(CX_K_SEQ); win_sum = 0; win_n = 0; have_prev = 0; logf2("coach: run live at chunk %lu, seq %lu", (unsigned long)last_chunk, (unsigned long)last_seq); }
		const uint64_t chunk = LD64(CX_K_CHUNK);
		if (chunk && chunk > last_chunk) {
			last_chunk = chunk;
			const double loss = kdbl(LD64(CX_K_LOSS)), lr = kdbl(LD64(CX_K_LR));
			logf2("report chunk %lu  loss %.4f nats = %.4f bits/token  lr %.3e  scale %.4f", (unsigned long)chunk, loss, loss / CX_LN2, lr, scale);
			if (force_scale > 0 && !forced) { scale = force_scale; ST64(CX_K_SCALE, kbits(scale)); forced = 1; logf2("coach: forced scale %.4f written", scale); }
			if (stop_at && (long)chunk >= stop_at) { ST64(CX_K_FLAGS, LD64(CX_K_FLAGS) | 2); logf2("coach: stop flag set at chunk %lu (--stop-at)", (unsigned long)chunk); }
			win_sum += loss; win_n++;
			if (win_n == window) {
				const double mean = win_sum / win_n;
				if (have_prev) {
					const double gain = (prev_mean - mean) / CX_LN2;
					logf2("window: mean %.4f bits/token, gain %.4f over the previous window", mean / CX_LN2, gain);
					if (gain < min_gain) {
						if (halvings >= max_halvings) { ST64(CX_K_FLAGS, LD64(CX_K_FLAGS) | 2); logf2("coach: plateau after %d halvings -> stop flag", halvings); }
						else { scale *= 0.5; halvings++; ST64(CX_K_SCALE, kbits(scale)); ST64(CX_K_FLAGS, LD64(CX_K_FLAGS) | 1); logf2("coach: plateau -> scale %.4f (halving %d), checkpoint requested", scale, halvings); }
					}
				}
				prev_mean = mean; have_prev = 1; win_sum = 0; win_n = 0;
			}
		}
		const uint64_t seq = LD64(CX_K_SEQ);
		if (seq != last_seq) {
			last_seq = seq;
			const uint32_t ns = LD32(CX_K_NSAMPLE);
			const volatile uint16_t *ids = (const volatile uint16_t *)(part + CX_K_SAMPLE);
			for (int p = 0; p < 2; p++) {
				char text[8192]; size_t len = 0;
				for (uint32_t i = 0; i < ns && i < 64; i++) {
					const uint16_t id = ids[p * 64 + i];
					if (id >= 256 + nmerges && id != EOT) continue;
					if (len + (size_t)tok_len[id] < sizeof text - 1) { memcpy(text + len, tok_s[id], (size_t)tok_len[id]); len += (size_t)tok_len[id]; }
				}
				text[len] = 0;
				for (size_t i = 0; i < len; i++) if (text[i] == '\n') text[i] = ' ';
				coh_metrics m;
				coh_score(text, len, &m);
				logf2("sample %lu prompt %d: hit5 %.3f  big %.3f  rep4 %.3f  words %ld | %s", (unsigned long)seq, p + 1, m.hit5, m.big, m.rep4, m.words, text);
				if (p == 0) {
					rep_streak = m.rep4 > rep4_band ? rep_streak + 1 : 0;
					if (rep_streak == 3) {
						rep_streak = 0;
						if (halvings < max_halvings) { scale *= 0.5; halvings++; ST64(CX_K_SCALE, kbits(scale)); logf2("coach: three samples with rep4 > %.3f -> scale %.4f (halving %d)", rep4_band, scale, halvings); }
					}
				}
			}
		}
	}
	return 0;
}
