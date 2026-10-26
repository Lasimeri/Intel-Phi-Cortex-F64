/*
 * train-check: Stage 3, host side. usage: train-check fwd|bwd DIR TOKENS.u16 L WARM
 * Replays the card's run (tools/train-gate.c) through tools/ref.h in double. fwd: compares DIR/fwd-loss.f32,
 * fwd-dl.f32, fwd-hc.f32. bwd: also gprm.f32 (per parameter group), gcg.f32, get.f32 (ETB layout) and dhc.f32:
 * largest absolute error and the error relative to the largest reference magnitude in that group.
 */
#include "ref.h"

#include <stdio.h>

static double *load(const char *dir, const char *name, size_t n)
{
	char path[4096];
	snprintf(path, sizeof path, "%s/%s", dir, name);
	FILE *f = fopen(path, "rb");
	double *v = malloc(n * sizeof *v);
	if (!f || fread(v, sizeof *v, n, f) != n) { perror(path); exit(1); }
	fclose(f);
	return v;
}

static void cmp(const char *name, const double *card, const double *ref, long n, long stride_card, long stride_ref)
{
	double amax = 0, rmax = 0;
	long at = -1;
	for (long i = 0; i < n; i++) {
		const double r = ref[i * stride_ref], d = fabs(card[i * stride_card] - r);
		if (d > amax) { amax = d; at = i; }
		if (fabs(r) > rmax) rmax = fabs(r);
	}
	printf("  %-6s n %7ld  max |ref| %.4e  max abs err %.3e  rel %.3e  (at %ld: card %.6e ref %.6e)\n", name, n, rmax, amax, rmax > 0 ? amax / rmax : 0, at,
	       at >= 0 ? card[at * stride_card] : 0, at >= 0 ? ref[at * stride_ref] : 0);
}

int main(int argc, char **argv)
{
	if (argc < 6 || (strcmp(argv[1], "fwd") && strcmp(argv[1], "bwd"))) { fprintf(stderr, "usage: train-check fwd|bwd DIR TOKENS.u16 L WARM\n"); return 2; }
	const int bwd = !strcmp(argv[1], "bwd");
	const char *dir = argv[2];
	const int L = atoi(argv[4]), warm = atoi(argv[5]);
	rparams P;
	rparams_alloc(&P);
	rparams_init(&P, 0);
	FILE *f = fopen(argv[3], "rb");
	if (!f) { perror(argv[3]); return 1; }
	uint16_t *tok = malloc((size_t)(warm + L + 1) * 2);
	if (fread(tok, 2, (size_t)(warm + L + 1), f) != (size_t)(warm + L + 1)) return 1;
	fclose(f);
	uint32_t *ids = malloc((size_t)(warm + L + 1) * 8);
	for (int t = 0; t < warm + L + 1; t++) ids[t] = tok[t];
	double *ring = malloc((long)CX_RING * CX_DB * CX_NBUS * 8);
	uint32_t dummy[CX_RING];
	cx_in_init(dummy, CX_RING, ring, 0);
	const double **bts = malloc((size_t)(warm + L) * sizeof *bts);
	for (int t = 0; t < warm + L; t++) bts[t] = ring + (long)(t % CX_RING) * CX_DB * CX_NBUS;

	double *T0 = calloc(CX_NT, sizeof *T0);
	rstep *steps = malloc((size_t)(L > warm ? L : warm) * sizeof *steps);
	for (int t = 0; t < warm; t++) ref_forward(&P, T0, &steps[t], ids[t], bts[t]);
	rgrads G;
	rgrads_alloc(&G);
	rgrads_zero(&G);
	const double loss = ref_chunk(&P, bwd ? &G : NULL, T0, ids + warm, bts + warm, L, steps);

	/* forward quantities */
	double *closs = load(dir, "fwd-loss.f32", (size_t)L), *cdl = load(dir, "fwd-dl.f32", (size_t)L * CX_VOCAB), *chc = load(dir, "fwd-hc.f32", (size_t)L * CX_DU);
	double *dl = malloc(CX_VOCAB * sizeof *dl), eloss = 0, edl = 0, dlmax = 0, ehc = 0, hcmax = 0, csum = 0;
	for (int t = 0; t < L; t++) {
		const double lt = ref_head(&P, &steps[t], ids[warm + t + 1], dl);
		csum += closs[t];
		if (fabs(lt - closs[t]) > eloss) eloss = fabs(lt - closs[t]);
		for (int v = 0; v < CX_VOCAB; v++) {
			const double r = dl[v] / L, d = fabs(cdl[(long)t * CX_VOCAB + v] - r);
			if (d > edl) edl = d;
			if (fabs(r) > dlmax) dlmax = fabs(r);
		}
		for (int c = 0; c < CX_DU; c++) {
			const double d = fabs(chc[(long)t * CX_DU + c] - steps[t].hc[c]);
			if (d > ehc) ehc = d;
			if (fabs(steps[t].hc[c]) > hcmax) hcmax = fabs(steps[t].hc[c]);
		}
	}
	printf("train-check %s: L %d warm %d: mean loss ref %.6f card %.6f; max |loss err| %.3e; dl max abs err %.3e (rel %.3e); hc max abs err %.3e (rel %.3e)\n",
	       argv[1], L, warm, loss, csum / L, eloss, edl, edl / dlmax, ehc, ehc / hcmax);
	if (!bwd) return 0;

	double *gprm = load(dir, "gprm.f32", CX_NPARAM), *gcg = load(dir, "gcg.f32", CX_DU), *get = load(dir, "get.f32", (size_t)CX_VOCAB * CX_DE), *cdhc = load(dir, "dhc.f32", (size_t)L * CX_DU);
	/* dhc: recompute from the reference (E^T dl / L per step) */
	double *rdhc = calloc((size_t)L * CX_DU, sizeof *rdhc);
	for (int t = 0; t < L; t++) {
		ref_head(&P, &steps[t], ids[warm + t + 1], dl);
		for (int v = 0; v < CX_VOCAB; v++) for (int c = 0; c < CX_DE; c++) rdhc[(long)t * CX_DU + c] += P.E[(long)v * CX_DE + c] * dl[v] / L;
	}
	printf("gradients (card F32 vs host double):\n");
	cmp("dhc", cdhc, rdhc, (long)L * CX_DU, 1, 1);
	cmp("P_in", gprm + CX_O_PIN, G.prm + CX_O_PIN, CX_DE * CX_DP, 1, 1);
	cmp("Wqb", gprm + CX_O_WQB, G.prm + CX_O_WQB, CX_DP * CX_DB, 1, 1);
	cmp("gain", gprm + CX_O_GAIN, G.prm + CX_O_GAIN, CX_DX, 1, 1);
	cmp("W", gprm + CX_O_W, G.prm + CX_O_W, CX_DX * CX_NW, 1, 1);
	cmp("W1", gprm + CX_O_W1, G.prm + CX_O_W1, CX_DM * CX_DH, 1, 1);
	cmp("b1", gprm + CX_O_B1, G.prm + CX_O_B1, CX_DH, 1, 1);
	cmp("Wu", gprm + CX_O_WU, G.prm + CX_O_WU, CX_DH * CX_DU, 1, 1);
	cmp("cg", gcg, G.cg, CX_DU, 1, 1);
	/* E: ETB layout -> row-major */
	double *ge = malloc((size_t)CX_VOCAB * CX_DE * sizeof *ge);
	for (int v = 0; v < CX_VOCAB; v++) for (int c = 0; c < CX_DE; c++) ge[(long)v * CX_DE + c] = get[(((long)v >> 4) * CX_DE + c) * 16 + (v & 15)];
	cmp("E", ge, G.E, (long)CX_VOCAB * CX_DE, 1, 1);
	double emax = 0;
	for (int t = 0; t <= L; t++) for (int c = 0; c < CX_DE; c++) { const long i = (long)ids[warm + t] * CX_DE + c; const double d = fabs(ge[i] - G.E[i]); if (d > emax) emax = d; }
	printf("  E rows used as inputs: max abs err %.3e\n", emax);
	return 0;
}
