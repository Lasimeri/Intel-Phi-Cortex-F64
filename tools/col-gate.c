/*
 * col-gate: Stage 1, card side. usage: col-gate OUTDIR STEPS THREADS [DUMP_THREAD...]
 *
 * THREADS columns (knc.h map, 2 per core) each run STEPS forward steps of card/col.h on their own weights and fast
 * weights, in lockstep (one core2r barrier per step, as cortexd will run) when THREADS > 1. Weights sit in a huge-page
 * arena at a distinct offset per thread (Stage 0f rule). Inputs come from lib/cx-col.h's generators (token ids and a
 * ring of 64 bus matrices per thread), which the host reference regenerates.
 *
 * Writes, for each DUMP_THREAD k (default 0): OUTDIR/col-nN-tK.f32 (STEPS x 160: u, bc, g) and -T.f32 (final 64x64);
 * OUTDIR/col-nN-ticks-tK.u32 (per-step TSC ticks of that thread); for THREADS > 1 OUTDIR/col-nN-step.u32 (thread 0,
 * barrier exit to barrier exit). Prints per-thread PMU counts per step: 0x2016 VPU instructions, 0x10cb L2 read
 * misses (card kernel arch/x86/events/intel/knc.c).
 */
#include "../card/team.h"
#include "../card/col.h"

#include <stdio.h>
#include <string.h>
#include <linux/perf_event.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#define ARENA (6u << 20)                     /* per thread: 2 MiB for the column + 4 MiB for its inputs */
#define COL_BYTES ((CX_NPARAM + CX_NT + CX_NACT) * 8)
#define RING_BYTES ((long)CX_RING * CX_DB * CX_NBUS * 8)   /* 1 MiB */

static int steps, nthr, ndump, ringn = CX_RING;
static uint64_t pmu0 = 0x0016, pmu1 = 0x10cb;   /* 0x0016 INSTRUCTIONS_EXECUTED (knc.c event map), 0x10cb L2_READ_MISS */
static int dump_list[CX_COLUMNS];
static double *E;
static char *arena[CX_COLUMNS];
static cx_col col[CX_COLUMNS];
static uint32_t *ids[CX_COLUMNS];
static double *ring[CX_COLUMNS];
static double *dump[CX_COLUMNS];
static uint32_t *ticks[CX_COLUMNS];
static uint32_t *step;
static uint64_t vpu[CX_COLUMNS], l2m[CX_COLUMNS];
static cx_barrier bar __attribute__((aligned(64)));
static double wall[2];

static int is_dumped(int k)
{
	for (int i = 0; i < ndump; i++) if (dump_list[i] == k) return 1;
	return 0;
}

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

static void setup(void *arg, int k)
{
	(void)arg;
	char *a = mmap(NULL, ARENA, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE, -1, 0);
	if (a == MAP_FAILED) return;
	arena[k] = a;
	/* a different 64-byte line per thread inside its first 2 MiB (Stage 0f: identical offsets evict each other) */
	const size_t lines = ((2u << 20) - COL_BYTES) / 64, off = ((size_t)k * 104729u % lines) * 64;
	col[k].prm = (double *)(a + off);
	col[k].T = col[k].prm + CX_NPARAM;
	col[k].act = col[k].T + CX_NT;
	cx_col_init(col[k].prm, k);
	memset(col[k].T, 0, CX_NT * 8);
	memset(col[k].act, 0, CX_NACT * 8);
	ring[k] = (double *)(a + (2u << 20));
	ids[k] = (uint32_t *)(a + (2u << 20) + RING_BYTES);
	cx_in_init(ids[k], steps, ring[k], k);
	ticks[k] = malloc((size_t)steps * 4);
	if (is_dumped(k)) dump[k] = aligned_alloc(64, ((size_t)steps * CX_DUMP + CX_NACT) * 8);   /* 512-bit stores need 64-byte alignment */
}

static void run(void *arg, int k)
{
	(void)arg;
	const int fv = counter(pmu0), fm = counter(pmu1);
	cx_barrier_wait(&bar, k, 1);
	uint64_t prev = cx_rdtsc();
	const uint64_t v0 = count(fv), m0 = count(fm);
	if (!k) wall[0] = cx_now();
	for (int t = 0; t < steps; t++) {
		const uint64_t t0 = cx_rdtsc();
		cx_col_step(&col[k], E + (long)ids[k][t] * CX_DE, ring[k] + (long)(t % ringn) * CX_DB * CX_NBUS);
		ticks[k][t] = (uint32_t)(cx_rdtsc() - t0);
		if (dump[k] && t == 0) memcpy(dump[k] + (long)steps * CX_DUMP, col[k].act, CX_NACT * 8);   /* the whole act block after step 0 */
		if (dump[k]) {
			double *o = dump[k] + (long)t * CX_DUMP;
			for (int v = 0; v < 16; v++) _mm512_store_pd(o + 8 * v, _mm512_load_pd(col[k].act + CX_A_U + 8 * v));
			_mm512_store_pd(o + 128, _mm512_load_pd(col[k].act + CX_A_BC));
			_mm512_store_pd(o + 136, _mm512_load_pd(col[k].act + CX_A_BC + 8));
			_mm512_store_pd(o + 144, _mm512_load_pd(col[k].act + CX_A_G));
			_mm512_store_pd(o + 152, _mm512_load_pd(col[k].act + CX_A_G + 8));
		}
		if (nthr > 1) {
			cx_barrier_wait(&bar, k, (uint64_t)t + 2);
			if (!k) { const uint64_t now = cx_rdtsc(); step[t] = (uint32_t)(now - prev); prev = now; }
		}
	}
	vpu[k] = count(fv) - v0;
	l2m[k] = count(fm) - m0;
	close(fv);
	close(fm);
	if (!k) wall[1] = cx_now();
}

int main(int argc, char **argv)
{
	if (argc < 4) { fprintf(stderr, "usage: col-gate OUTDIR STEPS THREADS [DUMP_THREAD...]\n"); return 2; }
	steps = atoi(argv[2]);
	nthr = atoi(argv[3]);
	if (steps < 1 || nthr < 1 || nthr > CX_COLUMNS) { fprintf(stderr, "col-gate: bad STEPS or THREADS\n"); return 2; }
	for (int i = 4; i < argc && ndump < CX_COLUMNS; i++) dump_list[ndump++] = atoi(argv[i]);
	if (!ndump) dump_list[ndump++] = 0;
	if (getenv("CX_RINGN")) ringn = atoi(getenv("CX_RINGN"));
	if (ringn < 1 || ringn > CX_RING) ringn = CX_RING;
	if (getenv("CX_STRIDE")) cx_stride = atoi(getenv("CX_STRIDE"));
	if (getenv("CX_CPU0")) cx_cpu0 = atoi(getenv("CX_CPU0"));
	if (cx_stride < 1 || nthr * cx_stride > CX_COLUMNS) cx_stride = 1;
	if (getenv("CX_PMU0")) pmu0 = strtoull(getenv("CX_PMU0"), NULL, 16);
	if (getenv("CX_PMU1")) pmu1 = strtoull(getenv("CX_PMU1"), NULL, 16);

	E = mmap(NULL, 4u << 20, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE, -1, 0);
	if (E == MAP_FAILED) { perror("embedding huge page"); return 1; }
	cx_emb_init(E);
	step = calloc((size_t)steps, 4);

	cx_team_run(nthr, setup, NULL);
	for (int k = 0; k < nthr; k++)
		if (!arena[k]) { fprintf(stderr, "col-gate: no huge pages for thread %d\n", k); return 1; }
	printf("col-gate: %d threads x %d steps, pf %d, ring %d, stride %d, pmu 0x%llx/0x%llx, column %zu B at %zu-B stride offsets, pin failures %d, thread 0 cpu %d, last cpu %d\n",
	       nthr, steps, CX_PF, ringn, cx_stride, (unsigned long long)pmu0, (unsigned long long)pmu1, (size_t)COL_BYTES, (size_t)104729 * 64, cx_pin_fail, cx_found_cpu[0], cx_found_cpu[nthr - 1]);

	cx_barrier_init(&bar, nthr, 2, CX_CORE_REL);
	cx_team_run(nthr, run, NULL);

	const double s = wall[1] - wall[0];
	printf("wall %.4f s: %.1f steps/s per thread, %.1f column-steps/s\n", s, steps / s, (double)steps * nthr / s);
	double vs = 0, ms = 0;
	uint64_t vmin = ~0ull, vmax = 0;
	for (int k = 0; k < nthr; k++) {
		vs += vpu[k]; ms += l2m[k];
		if (vpu[k] < vmin) vmin = vpu[k];
		if (vpu[k] > vmax) vmax = vpu[k];
	}
	printf("per step per thread: instructions (pmu0) %.1f (thread 0 %.1f, min %.1f, max %.1f), L2 read misses %.2f (thread 0 %.2f)\n",
	       vs / nthr / steps, (double)vpu[0] / steps, (double)vmin / steps, (double)vmax / steps, ms / nthr / steps, (double)l2m[0] / steps);
	/* thread 0 own-step ticks: min and a rough median from a 4096-bin histogram of the sorted copy is left to the host (stats) */
	uint32_t tmin = ~0u;
	double tsum = 0;
	for (int t = 0; t < steps; t++) { if (ticks[0][t] < tmin) tmin = ticks[0][t]; tsum += ticks[0][t]; }
	printf("thread 0 step ticks: min %u, mean %.1f (1238.093 MHz: %.2f us min, %.2f us mean)\n", tmin, tsum / steps, tmin / 1238.093, tsum / steps / 1238.093);

	char path[512];
	for (int i = 0; i < ndump; i++) {
		const int k = dump_list[i];
		if (k >= nthr) continue;
		snprintf(path, sizeof path, "%s/col-n%d-t%d.f32", argv[1], nthr, k);
		FILE *f = fopen(path, "wb");
		if (!f || fwrite(dump[k], 8, (size_t)steps * CX_DUMP + CX_NACT, f) != (size_t)steps * CX_DUMP + CX_NACT || fclose(f)) { perror(path); return 1; }
		snprintf(path, sizeof path, "%s/col-n%d-t%d-T.f32", argv[1], nthr, k);
		f = fopen(path, "wb");
		if (!f || fwrite(col[k].T, 8, CX_NT, f) != CX_NT || fclose(f)) { perror(path); return 1; }
		snprintf(path, sizeof path, "%s/col-n%d-ticks-t%d.u32", argv[1], nthr, k);
		f = fopen(path, "wb");
		if (!f || fwrite(ticks[k], 4, (size_t)steps, f) != (size_t)steps || fclose(f)) { perror(path); return 1; }
	}
	if (nthr > 1) {
		snprintf(path, sizeof path, "%s/col-n%d-step.u32", argv[1], nthr);
		FILE *f = fopen(path, "wb");
		if (!f || fwrite(step, 4, (size_t)steps, f) != (size_t)steps || fclose(f)) { perror(path); return 1; }
	}
	return 0;
}
