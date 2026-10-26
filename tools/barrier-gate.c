/*
 * barrier-gate: Stage 0d, card side. usage: barrier-gate OUTDIR [ROUNDS] [REPEATS]
 *
 * 120 pinned column threads (knc.h map) run ROUNDS back-to-back barriers per variant; thread 0 stores the TSC ticks
 * between consecutive barrier exits (one full barrier cycle: every arrival, then the release). Variants run
 * interleaved: for each repeat, every variant once. Raw tick deltas go to OUTDIR/bar-VARIANT-rREPEAT.u32 for the host
 * (build/host/stats); the first 1000 rounds of each run are warm-up and not written.
 */
#include "../card/team.h"

#include <stdio.h>
#include <string.h>

#define WARM 1000

typedef struct { const char *name; int arity, shape; } variant;
static const variant V[] = {
	{ "central", 0, CX_CENTRAL }, { "tree2", 2, CX_TREE }, { "tree4", 4, CX_TREE }, { "tree8", 8, CX_TREE },
	{ "flat", CX_COLUMNS - 1, CX_TREE }, { "tree4r", 4, CX_TREE_REL }, { "core4r", 4, CX_CORE_REL }, { "core2r", 2, CX_CORE_REL },
};
#define NV (int)(sizeof V / sizeof V[0])

static cx_barrier bar __attribute__((aligned(64)));
static uint32_t *ticks;
static long rounds;

static void run(void *arg, int k)
{
	(void)arg;
	uint64_t prev = cx_rdtsc();
	for (long r = 0; r < WARM + rounds; r++) {
		cx_barrier_wait(&bar, k, (uint64_t)r + 1);
		if (!k) {
			const uint64_t t = cx_rdtsc();
			if (r >= WARM) ticks[r - WARM] = (uint32_t)(t - prev);
			prev = t;
		}
	}
}

int main(int argc, char **argv)
{
	if (argc < 2) { fprintf(stderr, "usage: barrier-gate OUTDIR [ROUNDS] [REPEATS]\n"); return 2; }
	rounds = argc > 2 ? atol(argv[2]) : 100000;
	const int repeats = argc > 3 ? atoi(argv[3]) : 5;
	ticks = malloc((size_t)rounds * sizeof *ticks);
	for (int rep = 0; rep < repeats; rep++)
		for (int v = 0; v < NV; v++) {
			cx_barrier_init(&bar, CX_COLUMNS, V[v].arity, V[v].shape);
			const double t0 = cx_now();
			if (cx_team_run(CX_COLUMNS, run, NULL)) { fprintf(stderr, "barrier-gate: thread create failed\n"); return 1; }
			const double t1 = cx_now();
			char path[512];
			snprintf(path, sizeof path, "%s/bar-%s-r%d.u32", argv[1], V[v].name, rep);
			FILE *f = fopen(path, "wb");
			if (!f || fwrite(ticks, sizeof *ticks, (size_t)rounds, f) != (size_t)rounds || fclose(f)) { perror(path); return 1; }
			printf("rep %d %-8s %ld rounds in %.3f s wall\n", rep, V[v].name, rounds, t1 - t0);
			fflush(stdout);
		}
	return 0;
}
