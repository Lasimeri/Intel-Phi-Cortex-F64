/*
 * l2-gate: Stage 0f, card side. usage: l2-gate OUTDIR [REPEATS]
 *
 * 120 pinned column threads (knc.h map), all at once. Each runs y = x W over a K x N F32 block (K 411, N 128: 210,432
 * bytes, the size of one column's slow weights) with N in 8 vector accumulators: per k one broadcast of x[k] and 8
 * FMAs. Modes, interleaved within each repeat:
 *   (current modes: l2pf2, l2rpf2 (pseudo-random offset per thread), gddrpf, and lockpf2 / lockrpf2: the same with a
 *   barrier after every pass, thread 0 timing each step into -step.u32; argv[4] N runs only the first N threads)
 *   l2      - the same block every pass (L2 resident after the first pass; 2 threads per core = 411 KiB of 512)
 *   l2pf2   - as l2, plus vprefetch0 (L2 -> L1) of row k + 2 (all 8 lines of the row)
 *   l2st    - as l2, with the core's two blocks staggered in L2 set order (see stagger[])
 *   l2stpf2 - l2st plus vprefetch0 of row k + 2
 *   gddrpf  - a fresh block every pass from the thread's own 32 MiB of huge pages, vprefetch1 of row k + 16 and vprefetch0 of row k + 2
 * Thread k stores each pass's TSC ticks; OUTDIR/l2-MODE-rREP-all.u32 holds threads 0-117 and -c59.u32 threads 118-119
 * (core 59, beside the block pollers), for build/host/stats. Also prints aggregate GFLOP/s and GB/s per mode.
 */
#include "../card/team.h"

#include <immintrin.h>
#include <stdio.h>
#include <string.h>
#include <linux/perf_event.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#define K 411
#define N 128
#define BLOCK (K * N)                 /* floats */
#define BUF (32u << 20)               /* bytes per thread */
#define NBLK (int)(BUF / (BLOCK * 4)) /* 159 blocks */

#define NMODE 5
static int passes[NMODE] = { 4000, 4000, 4000, 4000, 300 };
/* per mode: 1 = a fresh block per pass from GDDR; prefetch distances in rows (0 = none) to L2 (vprefetch1) and L1 (vprefetch0) */
static const int fresh[NMODE] = { 0, 0, 0, 0, 1 }, pfl2[NMODE] = { 0, 0, 0, 0, 16 }, pfl1[NMODE] = { 2, 2, 2, 2, 2 };
/* lock: a barrier after every pass, as cortexd steps; thread 0 times each step (barrier exit to barrier exit) */
static const int lock[NMODE] = { 0, 1, 1, 0, 0 };
static uint32_t step[4000];
/* stagger: the core's second thread (k odd) puts its block right after the first's in L2 set order (offset BLOCK into its
 * own 2 MiB-aligned buffer), so the two blocks tile the 64 KiB-per-way sets instead of piling on the same ones */
static const int stagger[NMODE] = { 0, 0, 2, 2, 0 };   /* 2: pseudo-random offset per thread, see run() */
static const char *mode_name[NMODE] = { "l2pf2", "lockpf2", "lockrpf2", "l2rpf2", "gddrpf" };

static float *buf[CX_COLUMNS];
static uint32_t *ticks[CX_COLUMNS];
static float sink[CX_COLUMNS][16] __attribute__((aligned(64)));
static cx_barrier bar __attribute__((aligned(64)));
static int mode;
static double wall[2];

static inline void pf0(const void *p) { __asm__ volatile("vprefetch0 %0" : : "m"(*(const char *)p)); }
static inline void pf1(const void *p) { __asm__ volatile("vprefetch1 %0" : : "m"(*(const char *)p)); }

static inline __m512 bcast(const float *p) { return _mm512_extload_ps(p, _MM_UPCONV_PS_NONE, _MM_BROADCAST_1X16, _MM_HINT_NONE); }

static void gemv(const float *W, const float *x, float *y, int d2, int d1)
{
	__m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
	for (int k = 0; k < K; k++) {
		const float *w = W + k * N;
		if (d2) for (int l = 0; l < 8; l++) pf1(w + d2 * N + 16 * l);
		if (d1) {
			for (int l = 0; l < 8; l++) pf0(w + d1 * N + 16 * l);
		}
		const __m512 xb = bcast(x + k);
		a0 = _mm512_fmadd_ps(xb, _mm512_load_ps(w), a0);
		a1 = _mm512_fmadd_ps(xb, _mm512_load_ps(w + 16), a1);
		a2 = _mm512_fmadd_ps(xb, _mm512_load_ps(w + 32), a2);
		a3 = _mm512_fmadd_ps(xb, _mm512_load_ps(w + 48), a3);
		a4 = _mm512_fmadd_ps(xb, _mm512_load_ps(w + 64), a4);
		a5 = _mm512_fmadd_ps(xb, _mm512_load_ps(w + 80), a5);
		a6 = _mm512_fmadd_ps(xb, _mm512_load_ps(w + 96), a6);
		a7 = _mm512_fmadd_ps(xb, _mm512_load_ps(w + 112), a7);
	}
	a0 = _mm512_add_ps(_mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3)), _mm512_add_ps(_mm512_add_ps(a4, a5), _mm512_add_ps(a6, a7)));
	_mm512_store_ps(y, a0);
}

static void setup(void *arg, int k)
{
	(void)arg;
	buf[k] = mmap(NULL, BUF + (2u << 20), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE, -1, 0);
	if (buf[k] == MAP_FAILED) { buf[k] = NULL; return; }
	const __m512 v = _mm512_set1_ps(1.0f / 1024);
	for (size_t i = 0; i < (BUF + (2u << 20)) / 4; i += 16) _mm512_store_ps(buf[k] + i, v);
}

/* per-thread PMU counts over one mode's passes: KNC raw events from the card kernel's arch/x86/events/intel/knc.c
 * (0x10cb L2_READ_MISS, 0x0003 DATA_READ_MISS = L1 data read miss); KNC has two counters per thread */
static uint64_t l2miss[CX_COLUMNS], l1miss[CX_COLUMNS];

static int counter(uint64_t config)
{
	struct perf_event_attr pa;
	memset(&pa, 0, sizeof pa);
	pa.size = sizeof pa;
	pa.type = PERF_TYPE_RAW;
	pa.config = config;
	pa.exclude_kernel = 1;
	pa.exclude_hv = 1;
	return (int)syscall(__NR_perf_event_open, &pa, 0, -1, -1, 0);
}

static uint64_t count(int fd)
{
	uint64_t v = 0;
	return fd >= 0 && read(fd, &v, sizeof v) == sizeof v ? v : 0;
}

static void run(void *arg, int k)
{
	(void)arg;
	static float x[K + 16] __attribute__((aligned(64)));
	const int f2 = counter(0x10cb), f1 = counter(0x0003);
	cx_barrier_wait(&bar, k, 1);
	uint64_t prev = cx_rdtsc();
	const uint64_t m2 = count(f2), m1 = count(f1);
	/* a different 64-byte-aligned offset per thread inside its 32 MiB: line (k * 104729) mod 2^19, as floats */
	const size_t roff = ((size_t)k * 104729u % (BUF / 64 - BLOCK / 16)) * 16;
	if (!k) wall[0] = cx_now();
	const int n = passes[mode];
	for (int p = 0; p < n; p++) {
		const float *W = buf[k] + (fresh[mode] ? (size_t)(p % NBLK) * BLOCK : stagger[mode] == 2 ? roff : stagger[mode] && k % 2 ? (size_t)BLOCK : 0);
		const uint64_t t0 = cx_rdtsc();
		gemv(W, x, sink[k], pfl2[mode], pfl1[mode]);
		ticks[k][p] = (uint32_t)(cx_rdtsc() - t0);
		if (lock[mode]) {
			cx_barrier_wait(&bar, k, (uint64_t)p + 2);
			if (!k) { const uint64_t t = cx_rdtsc(); step[p] = (uint32_t)(t - prev); prev = t; }
		}
	}
	l2miss[k] = count(f2) - m2;
	l1miss[k] = count(f1) - m1;
	close(f2);
	close(f1);
	cx_barrier_wait(&bar, k, lock[mode] ? (uint64_t)n + 2 : 2);
	if (!k) wall[1] = cx_now();
}

/* threads on core 59 (CPUs 237-240) go to -c59, the rest to -all */
static int dump(const char *dir, int rep, int n, int c59, const char *tag)
{
	char path[512];
	snprintf(path, sizeof path, "%s/l2-%s-n%d-r%d-%s.u32", dir, mode_name[mode], n, rep, tag);
	FILE *f = fopen(path, "wb");
	if (!f) { perror(path); return -1; }
	for (int k = 0; k < n; k++)
		if ((cx_found_cpu[k] >= 237 && cx_found_cpu[k] <= 240) == c59) fwrite(ticks[k], 4, (size_t)passes[mode], f);
	return fclose(f);
}

/* pinning check after a team run: failures, and threads not on the CPU the map gives */
static void pin_report(int n, const char *when)
{
	int off = 0;
	for (int k = 0; k < n; k++) off += cx_found_cpu[k] != cx_cpu(k * cx_stride);
	printf("pinning (%s): %d threads, %d setaffinity failures, %d on another CPU; thread 0 cpu %d, thread %d cpu %d\n", when, n,
	       cx_pin_fail, off, cx_found_cpu[0], n - 1, cx_found_cpu[n - 1]);
}

int main(int argc, char **argv)
{
	if (argc < 2) { fprintf(stderr, "usage: l2-gate OUTDIR [REPEATS] [STRIDE]\n"); return 2; }
	const int repeats = argc > 2 ? atoi(argv[2]) : 3;
	cx_stride = argc > 3 ? atoi(argv[3]) : 1;
	if (cx_stride < 1 || CX_COLUMNS % cx_stride) { fprintf(stderr, "l2-gate: stride must divide %d\n", CX_COLUMNS); return 2; }
	const int n = argc > 4 ? atoi(argv[4]) : CX_COLUMNS / cx_stride;   /* N: the first N threads of the map (118 = cores 0-58) */
	for (int k = 0; k < n; k++) ticks[k] = malloc(4000 * sizeof(uint32_t));
	cx_team_run(n, setup, NULL);
	pin_report(n, "setup");
	for (int k = 0; k < n; k++)
		if (!buf[k]) { fprintf(stderr, "l2-gate: no huge pages for thread %d\n", k); return 1; }
	for (int rep = 0; rep < repeats; rep++)
		for (mode = 0; mode < NMODE; mode++) {
			cx_barrier_init(&bar, n, 2, CX_CORE_REL);
			cx_team_run(n, run, NULL);
			const double s = wall[1] - wall[0], flop = 2.0 * K * N * passes[mode] * n;
			printf("rep %d %-7s %4d passes x %d threads in %.4f s: %.1f GFLOP/s, %.1f GB/s of weights\n", rep, mode_name[mode],
			       passes[mode], n, s, flop / s / 1e9, (double)BLOCK * 4 * passes[mode] * n / s / 1e9);
			uint64_t s2 = 0, s1 = 0, max2 = 0;
			for (int k = 0; k < n; k++) {
				s2 += l2miss[k];
				s1 += l1miss[k];
				if (l2miss[k] > max2) max2 = l2miss[k];
			}
			printf("    per pass per thread: L2 read misses %.1f (worst thread %.1f), L1 data read misses %.1f; lines per block %d\n",
			       (double)s2 / n / passes[mode], (double)max2 / passes[mode], (double)s1 / n / passes[mode], BLOCK * 4 / 64);
			fflush(stdout);
			if (dump(argv[1], rep, n, 0, "all") || dump(argv[1], rep, n, 1, "c59")) return 1;
			if (lock[mode]) {
				char path[512];
				snprintf(path, sizeof path, "%s/l2-%s-n%d-r%d-step.u32", argv[1], mode_name[mode], n, rep);
				FILE *f = fopen(path, "wb");
				if (!f || fwrite(step, 4, (size_t)passes[mode], f) != (size_t)passes[mode] || fclose(f)) { perror(path); return 1; }
			}
		}
	pin_report(n, "last run");
	return 0;
}
