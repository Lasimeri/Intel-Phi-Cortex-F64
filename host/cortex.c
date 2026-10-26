/*
 * cortex: the host side of the Intel Phi Cortex (Stage 5). usage:
 *   cortex CKPT "prompt" NGEN [--greedy] [--bus 0|1] [--temp T] [--topk K] [--topp P] [--minp M] [--rp R] [--seed S] [--vocab FILE]
 *
 * Talks to card/cortexd through the mailbox in the shared host-memory region (lib/cx-proto.h): stages the checkpoint,
 * LOAD, BUS, RESET, then one STEP per token. The card answers each STEP with the 4096 raw logits by DMA; this program
 * samples them in llama.cpp's default order (repeat penalty over the last 64 tokens, top-k, top-p, min-p,
 * temperature, draw) with the user's settings (temp 0.5, top-k 75, top-p 0.95, min-p 0.05, rp 1.2) and streams the
 * detokenised text. --greedy takes the argmax instead (the card-side baseline, byte-identical ids).
 */
#define _GNU_SOURCE
#define BPE_NO_MAIN
#include "../tools/bpe.c"
#include "../lib/cx-proto.h"
#include "../lib/cx-col.h"

#include <fcntl.h>
#include <math.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

static uint8_t *part;
#define LD64(o) __atomic_load_n((uint64_t *)(part + (o)), __ATOMIC_ACQUIRE)
#define LD32(o) __atomic_load_n((uint32_t *)(part + (o)), __ATOMIC_ACQUIRE)
#define ST64(o, x) __atomic_store_n((uint64_t *)(part + (o)), (x), __ATOMIC_RELEASE)
#define ST32(o, x) __atomic_store_n((uint32_t *)(part + (o)), (x), __ATOMIC_RELEASE)

static uint64_t ns(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return (uint64_t)t.tv_sec * 1000000000u + (uint64_t)t.tv_nsec;
}

static uint64_t seq;

static int command(uint32_t op, uint32_t arg, double timeout_s)
{
	seq++;
	ST32(CX_H_CMD_OP, op);
	ST64(CX_H_CMD, CX_CMD(seq, arg));
	const uint64_t t0 = ns();
	while (LD64(CX_C_ACK) != seq) {
		if (LD32(CX_C_STATE) == 2) { fprintf(stderr, "cortex: the card reported an error on op %u\n", op); return -1; }
		if (ns() - t0 > (uint64_t)(timeout_s * 1e9)) { fprintf(stderr, "cortex: op %u timed out after %.1f s\n", op, timeout_s); return -1; }
	}
	return 0;
}

typedef struct { int id; double l; double p; } cand;
static int cmp_l(const void *a, const void *b) { const double x = ((const cand *)a)->l, y = ((const cand *)b)->l; return (x < y) - (x > y); }

static uint64_t rng;
static double uni(void) { return (double)(cx_rng(&rng) >> 11) / 9007199254740992.0; }

/* the sampler: logits in, token out */
static int sample(const double *logits, const uint32_t *hist, int nhist, int greedy, float temp, int topk, float topp, float minp, float rp)
{
	static cand c[CX_VOCAB];
	for (int i = 0; i < CX_VOCAB; i++) { c[i].id = i; c[i].l = logits[i]; }
	if (greedy) {
		int best = 0;
		for (int i = 1; i < CX_VOCAB; i++) if (c[i].l > c[best].l) best = i;
		return best;
	}
	/* repeat penalty over the last 64 tokens (llama.cpp: positive logits divided, negative multiplied) */
	for (int i = nhist > 64 ? nhist - 64 : 0; i < nhist; i++) {
		cand *x = &c[hist[i]];
		x->l = x->l > 0 ? x->l / rp : x->l * rp;
	}
	qsort(c, CX_VOCAB, sizeof *c, cmp_l);
	int n = topk < CX_VOCAB ? topk : CX_VOCAB;
	/* softmax over the kept candidates, then top-p, min-p */
	double sum = 0;
	for (int i = 0; i < n; i++) { c[i].p = exp((double)c[i].l - c[0].l); sum += c[i].p; }
	for (int i = 0; i < n; i++) c[i].p /= sum;
	double cum = 0;
	int m = n;
	for (int i = 0; i < n; i++) { cum += c[i].p; if (cum >= topp) { m = i + 1; break; } }
	n = m;
	m = 0;
	for (int i = 0; i < n; i++) if (c[i].p >= minp * c[0].p) m = i + 1;
	n = m > 0 ? m : 1;
	/* temperature on the logits of the survivors, re-normalise, draw */
	sum = 0;
	for (int i = 0; i < n; i++) { c[i].p = exp(((double)c[i].l - c[0].l) / temp); sum += c[i].p; }
	double r = uni() * sum;
	for (int i = 0; i < n; i++) { r -= c[i].p; if (r <= 0) return c[i].id; }
	return c[n - 1].id;
}

int main(int argc, char **argv)
{
	if (argc >= 2 && !strcmp(argv[1], "--stop")) {
		int sfd = open("/dev/shm/phi-hostmem", O_RDWR);
		if (sfd < 0) { perror("/dev/shm/phi-hostmem"); return 1; }
		part = mmap(NULL, CX_PART_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, sfd, (off_t)CX_PART_MIB << 20);
		close(sfd);
		ST32(CX_H_MAGIC, CX_MAGIC);
		ST32(CX_H_VERSION, CX_VERSION);
		seq = CX_CMD_SEQ(LD64(CX_H_CMD));
		if (seq == CX_CMD_QUIT) seq = 0;
		ST64(CX_H_CMD, CX_CMD(seq, 0));
		const uint64_t s2 = ns();
		ST64(CX_H_SESSION, s2);
		const uint64_t w0 = ns();
		while (LD64(CX_C_SESSION) != s2 || LD32(CX_C_STATE) != 1) { if (ns() - w0 > 10000000000ull) { fprintf(stderr, "cortex: no daemon took the session in 10 s\n"); return 1; } usleep(1000); }
		if (command(CX_OP_EXIT, 0, 5)) return 1;
		fprintf(stderr, "cortex: daemon stopped\n");
		return 0;
	}
	if (argc < 4) { fprintf(stderr, "usage: cortex --stop | cortex CKPT \"prompt\" NGEN [--greedy] [--bus 0|1] [--temp T] [--topk K] [--topp P] [--minp M] [--rp R] [--seed S] [--vocab FILE]\n"); return 2; }
	const char *ckpt = argv[1], *vocab = "data/bpe/vocab4096.txt";
	char *prompt = argv[2];
	int ngen = atoi(argv[3]);
	int greedy = 0, bus = 1, topk = 75, chat = 0;
	float temp = 0.5f, topp = 0.95f, minp = 0.05f, rp = 1.2f;
	rng = 0x5EED0000C0A7E8ull;
	for (int i = 4; i < argc; i++) {
		if (!strcmp(argv[i], "--greedy")) greedy = 1;
		else if (!strcmp(argv[i], "--chat")) chat = 1;
		else if (!strcmp(argv[i], "--bus") && i + 1 < argc) bus = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--temp") && i + 1 < argc) temp = (float)atof(argv[++i]);
		else if (!strcmp(argv[i], "--topk") && i + 1 < argc) topk = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--topp") && i + 1 < argc) topp = (float)atof(argv[++i]);
		else if (!strcmp(argv[i], "--minp") && i + 1 < argc) minp = (float)atof(argv[++i]);
		else if (!strcmp(argv[i], "--rp") && i + 1 < argc) rp = (float)atof(argv[++i]);
		else if (!strcmp(argv[i], "--seed") && i + 1 < argc) rng = strtoull(argv[++i], NULL, 10);
		else if (!strcmp(argv[i], "--vocab") && i + 1 < argc) vocab = argv[++i];
	}
	if (chat) { char *w = malloc(strlen(prompt) + 40); sprintf(w, "\n\nHuman: %s\n\nAssistant:", prompt); prompt = w; }
	if (load_vocab(vocab)) return 1;
	for (int id = 0; id < 256 + nmerges; id++) expand(id);
	expand(EOT);

	/* the prompt's tokens */
	uint32_t *ids = malloc((size_t)(strlen(prompt) + ngen + 16) * 4);
	int nid = 0;
	{
		const unsigned char *p = (const unsigned char *)prompt;
		long n = (long)strlen(prompt), pos = 0;
		uint16_t tmp[WMAX];
		while (pos < n) {
			int sp;
			const int len = next_word(p + pos, n - pos, &sp);
			if (sp) ids[nid++] = EOT;
			else { const int m = encode_word(p + pos, len, tmp); for (int i = 0; i < m; i++) ids[nid++] = tmp[i]; }
			pos += len;
		}
	}
	const int nprompt = nid;

	/* the mailbox session */
	int fd = open("/dev/shm/phi-hostmem", O_RDWR);
	if (fd < 0) { perror("/dev/shm/phi-hostmem"); return 1; }
	part = mmap(NULL, CX_PART_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)CX_PART_MIB << 20);
	close(fd);
	if (part == MAP_FAILED) { perror("mmap"); return 1; }
	ST32(CX_H_MAGIC, CX_MAGIC);
	ST32(CX_H_VERSION, CX_VERSION);
	seq = CX_CMD_SEQ(LD64(CX_H_CMD));
	if (seq == CX_CMD_QUIT) seq = 0;
	ST64(CX_H_CMD, CX_CMD(seq, 0));
	const uint64_t session = ns();
	ST64(CX_H_SESSION, session);
	uint64_t t0 = ns();
	while (LD64(CX_C_SESSION) != session || LD32(CX_C_STATE) != 1) {
		if (ns() - t0 > 30000000000ull) { fprintf(stderr, "cortex: cortexd did not take the session in 30 s (is it running on the card?)\n"); return 1; }
		usleep(1000);
	}
	/* stage and load the checkpoint */
	FILE *f = fopen(ckpt, "rb");
	if (!f) { perror(ckpt); return 1; }
	const size_t cbytes = ((size_t)CX_NCOL * CX_NPARAM + CX_DU + (size_t)CX_VOCAB * CX_DE) * 8;
	if (fread(part + CX_STAGE, 1, cbytes, f) != cbytes) { fprintf(stderr, "cortex: short checkpoint %s\n", ckpt); return 1; }
	fclose(f);
	__atomic_thread_fence(__ATOMIC_SEQ_CST);
	t0 = ns();
	if (command(CX_OP_LOAD, 0, 30)) return 1;
	fprintf(stderr, "cortex: checkpoint loaded by DMA in %.1f ms; prompt %d tokens; %s\n", (ns() - t0) / 1e6, nprompt, greedy ? "greedy" : "sampling");
	if (command(CX_OP_BUS, (uint32_t)bus, 5) || command(CX_OP_RESET, 0, 5)) return 1;

	fwrite(prompt, 1, strlen(prompt), stdout);
	fflush(stdout);
	double *logits = malloc(CX_VOCAB * sizeof *logits);
	const uint64_t g0 = ns();
	uint64_t card_ns = 0, host_ns = 0;
	for (int s = 0; s < nprompt + ngen - 1; s++) {
		const uint64_t a = ns();
		if (command(CX_OP_STEP, ids[s], 5)) return 1;
		const uint64_t b = ns();
		card_ns += b - a;
		if (s + 1 < nprompt) continue;
		memcpy(logits, part + CX_LOGITS, CX_VOCAB * sizeof *logits);
		const int id = sample(logits, ids, s + 1, greedy, temp, topk, topp, minp, rp);
		if (chat && id == EOT) { ngen = s + 1 - nprompt; break; }
		ids[s + 1] = (uint32_t)id;
		if (chat) {
			static char tail[64]; static int tl;
			const int add = tok_len[id] < 63 ? tok_len[id] : 63;
			if (tl + add > 63) { memmove(tail, tail + tl + add - 63, (size_t)(63 - add)); tl = 63 - add; }
			memcpy(tail + tl, tok_s[id], (size_t)add); tl += add; tail[tl] = 0;
			if (strstr(tail, "\n\nHuman:") || strstr(tail, "\nHuman:")) { ngen = s + 1 - nprompt; break; }
		}
		fwrite(tok_s[id], 1, (size_t)tok_len[id], stdout);
		fflush(stdout);
		host_ns += ns() - b;
	}
	const uint64_t g1 = ns();
	printf("\n");
	fprintf(stderr, "cortex: %d prompt + %d generated tokens in %.3f s: %.1f tok/s overall; per token card %.1f us (mailbox + forward + DMA), host %.1f us (sampling + print)\n",
	        nprompt, ngen, (g1 - g0) / 1e9, (nprompt + ngen - 1) / ((g1 - g0) / 1e9), card_ns / 1e3 / (nprompt + ngen - 1), host_ns / 1e3 / ngen);
	ST64(CX_H_CMD, CX_CMD(CX_CMD_QUIT, 0));
	f = fopen("data/cortex-last.u16", "wb");   /* the ids of the last run, for cmp against a baseline */
	if (f) { for (int i = 0; i < nprompt + ngen; i++) { const uint16_t v = (uint16_t)ids[i]; fwrite(&v, 2, 1, f); } fclose(f); }
	return 0;
}
