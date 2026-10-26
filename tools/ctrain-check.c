/*
 * ctrain-check: Stage 4, host side. usage: ctrain-check fwd|bwd DIR TOKENS.u16 L WARM [DUMP_COL...]
 * Replays the card's cluster run (tools/cluster-gate.c) through tools/cref.h (120 columns, double, live bus from
 * zero) and compares: per-step loss, hc, and u / bc / g of the dumped columns; bwd: their parameter gradients, cg, E.
 */
#include "cref.h"

#include <stdio.h>

static double *load(const char *dir, const char *name, int k, size_t n)
{
	char path[4096];
	if (k >= 0) snprintf(path, sizeof path, "%s/%s%d.f32", dir, name, k);
	else snprintf(path, sizeof path, "%s/%s.f32", dir, name);
	FILE *f = fopen(path, "rb");
	double *v = malloc(n * sizeof *v);
	if (!f || fread(v, sizeof *v, n, f) != n) { fprintf(stderr, "ctrain-check: %s missing or short, skipped\n", path); if (f) fclose(f); free(v); return NULL; }
	fclose(f);
	return v;
}

static void cmp(const char *name, int k, const double *card, const double *ref, long n)
{
	double amax = 0, rmax = 0;
	long at = -1;
	for (long i = 0; i < n; i++) {
		const double d = fabs(card[i] - ref[i]);
		if (d > amax) { amax = d; at = i; }
		if (fabs(ref[i]) > rmax) rmax = fabs(ref[i]);
	}
	printf("  %-5s col %3d  n %7ld  max |ref| %.4e  max abs err %.3e  rel %.3e  (at %ld: card %.6e ref %.6e)\n", name, k, n, rmax, amax,
	       rmax > 0 ? amax / rmax : 0, at, at >= 0 ? card[at] : 0, at >= 0 ? ref[at] : 0);
}

int main(int argc, char **argv)
{
	if (argc < 6) { fprintf(stderr, "usage: ctrain-check fwd|bwd DIR TOKENS.u16 L WARM [DUMP_COL...]\n"); return 2; }
	const int bwd = !strcmp(argv[1], "bwd"), L = atoi(argv[4]), warm = atoi(argv[5]);
	const char *dir = argv[2];
	int ndump = 0, dl_[CX_NCOL];
	for (int i = 6; i < argc; i++) dl_[ndump++] = atoi(argv[i]);
	if (!ndump) { dl_[ndump++] = 0; dl_[ndump++] = 59; dl_[ndump++] = 119; }
	const int N = CX_NCOL;
	cref C;
	cref_alloc(&C, N, L);
	cref_init(&C);
	cgrads G;
	cgrads_alloc(&G, N);
	cgrads_zero(&G, N);
	FILE *f = fopen(argv[3], "rb");
	if (!f) { perror(argv[3]); return 1; }
	uint16_t *tok = malloc((size_t)(warm + L + 1) * 2);
	if (fread(tok, 2, (size_t)(warm + L + 1), f) != (size_t)(warm + L + 1)) return 1;
	fclose(f);
	uint32_t *ids = malloc((size_t)(warm + L + 1) * 8);
	for (int t = 0; t < warm + L + 1; t++) ids[t] = tok[t];

	/* warm-up chunks (live bus, state carried), then the chunk */
	C.bus_mode = getenv("CX_BUS") && atoi(getenv("CX_BUS")) == 0 ? CREF_BUS_ZERO : CREF_BUS_LIVE;
	for (int ch = 0; ch < warm / L; ch++) {
		for (int t = 0; t < L; t++) cref_step(&C, t, ids[(long)ch * L + t]);
		/* the next chunk's bus[0] = this chunk's bus[L] */
		memcpy(C.bus, C.bus + (long)L * CX_DB * CX_NBUS, CX_DB * CX_NBUS * sizeof *C.bus);
	}
	double *T0 = malloc((size_t)N * CX_NT * sizeof *T0);
	double *bus0 = malloc(CX_DB * CX_NBUS * sizeof *bus0);
	memcpy(T0, C.T, (size_t)N * CX_NT * sizeof *T0);
	memcpy(bus0, C.bus, CX_DB * CX_NBUS * sizeof *bus0);
	const double loss = cref_chunk(&C, bwd ? &G : NULL, T0, bus0, ids + warm, L);

	double *closs = load(dir, "cl-loss", -1, (size_t)L), *chc = load(dir, "cl-hc", -1, (size_t)L * CX_DU);
	if (!closs || !chc) return 1;
	double *dlv = malloc(CX_VOCAB * sizeof *dlv), eloss = 0, csum = 0;
	double *rhc = malloc((size_t)(L > 0 ? L : 0) * CX_DU * sizeof *rhc);
	for (int t = 0; t < L; t++) {
		const double lt = ref_head_hc(C.E, C.cs[t].hc, ids[warm + t + 1], dlv);
		csum += closs[t];
		if (fabs(lt - closs[t]) > eloss) eloss = fabs(lt - closs[t]);
		for (int c = 0; c < CX_DU; c++) rhc[(long)t * CX_DU + c] = C.cs[t].hc[c];
	}
	printf("ctrain-check %s: N %d, L %d, warm %d: mean loss ref %.6f card %.6f; max |loss err| %.3e\n", argv[1], N, L, warm, loss, csum / L, eloss);
	cmp("hc", -1, chc, rhc, (long)L * CX_DU);
	for (int i = 0; i < ndump; i++) {
		const int k = dl_[i];
		double *cu = load(dir, "cl-u", k, (size_t)(L > 0 ? L : 0) * CX_DUMP);
		if (!cu) continue;
		double *ru = malloc((size_t)L * CX_DU * sizeof *ru), *rb = malloc((size_t)L * 16 * sizeof *rb), *rg = malloc((size_t)L * 16 * sizeof *rg);
		double *cu2 = malloc((size_t)L * CX_DU * sizeof *cu2), *cb = malloc((size_t)L * 16 * sizeof *cb), *cgv = malloc((size_t)L * 16 * sizeof *cgv);
		for (int t = 0; t < L; t++) {
			const rstep *r = &C.steps[(long)k * L + t];
			for (int c = 0; c < CX_DU; c++) { ru[(long)t * CX_DU + c] = r->u[c]; cu2[(long)t * CX_DU + c] = cu[(long)t * CX_DUMP + c]; }
			for (int d = 0; d < 16; d++) { rb[t * 16 + d] = r->bc[d]; cb[t * 16 + d] = cu[(long)t * CX_DUMP + 128 + d]; }
			for (int d = 0; d < 8; d++) { rg[t * 8 + d] = r->g; cgv[t * 8 + d] = cu[(long)t * CX_DUMP + 144 + d]; }
		}
		cmp("u", k, cu2, ru, (long)L * CX_DU);
		cmp("bc", k, cb, rb, (long)L * 16);
		cmp("g", k, cgv, rg, (long)L * 8);
		if (bwd) {
			double *gp = load(dir, "cl-gprm", k, CX_NPARAM);
			if (!gp) continue;
			const double *rp = G.prm + (long)k * CX_NPARAM;
			cmp("P_in", k, gp + CX_O_PIN, rp + CX_O_PIN, CX_DE * CX_DP);
			cmp("Wqb", k, gp + CX_O_WQB, rp + CX_O_WQB, CX_DP * CX_DB);
			cmp("gain", k, gp + CX_O_GAIN, rp + CX_O_GAIN, CX_DX);
			cmp("W", k, gp + CX_O_W, rp + CX_O_W, CX_DX * CX_NW);
			cmp("W1", k, gp + CX_O_W1, rp + CX_O_W1, CX_DM * CX_DH);
			cmp("b1", k, gp + CX_O_B1, rp + CX_O_B1, CX_DH);
			cmp("Wu", k, gp + CX_O_WU, rp + CX_O_WU, CX_DH * CX_DU);
			cmp("wg", k, gp + CX_O_WG, rp + CX_O_WG, CX_DH);
			cmp("bg", k, gp + CX_O_BG, rp + CX_O_BG, 1);
		}
	}
	if (bwd) {
		double *gcg = load(dir, "cl-gcg", -1, CX_DU), *get = load(dir, "cl-get", -1, (size_t)CX_VOCAB * CX_DE);
		if (!gcg || !get) return 0;
		cmp("cg", -1, gcg, G.cg, CX_DU);
		double *ge = malloc((size_t)CX_VOCAB * CX_DE * sizeof *ge);
		for (int v = 0; v < CX_VOCAB; v++) for (int c = 0; c < CX_DE; c++) ge[(long)v * CX_DE + c] = get[(((long)v >> 4) * CX_DE + c) * 16 + (v & 15)];
		cmp("E", -1, ge, G.E, (long)CX_VOCAB * CX_DE);
	}
	return 0;
}
