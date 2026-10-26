/*
 * train-ref: Stage 3a. Finite-difference check of tools/ref.h (the host double reference of one column's training
 * step) on sampled parameters of every group. usage: train-ref [L] [WARMUP] [SAMPLES]
 * Tokens come from data/bpe/valid.u16 (first WARMUP + L + 1 tokens); the bus inputs from the Stage 1 ring (column 0).
 * Prints, per group, the largest |fd - grad| / max(|fd|, |grad|, 1e-10) over SAMPLES entries, the largest |grad|,
 * and the loss. Central differences with h = 1e-6 in double.
 */
#include "ref.h"

#include <stdio.h>

static double *slot(rparams *P, int group, long idx)
{
	switch (group) {
	case 0: return P->prm + CX_O_PIN + idx;
	case 1: return P->prm + CX_O_WQB + idx;
	case 2: return P->prm + CX_O_GAIN + idx;
	case 3: return P->prm + CX_O_W + idx;
	case 4: return P->prm + CX_O_W1 + idx;
	case 5: return P->prm + CX_O_B1 + idx;
	case 6: return P->prm + CX_O_WU + idx;
	case 7: return P->cg + idx;
	default: return P->E + idx;
	}
}

static const char *gname[] = { "P_in", "Wqb", "gain", "W", "W1", "b1", "Wu", "cg", "E" };
static const long gsize[] = { CX_DE * CX_DP, CX_DP * CX_DB, CX_DX, CX_DX * CX_NW, CX_DM * CX_DH, CX_DH, CX_DH * CX_DU, CX_DU, (long)CX_VOCAB * CX_DE };

int main(int argc, char **argv)
{
	const int L = argc > 1 ? atoi(argv[1]) : 8, warm = argc > 2 ? atoi(argv[2]) : 8, samples = argc > 3 ? atoi(argv[3]) : 24;
	rparams P;
	rparams_alloc(&P);
	rparams_init(&P, 0);
	rgrads G;
	rgrads_alloc(&G);

	FILE *f = fopen("data/bpe/valid.u16", "rb");
	if (!f) { perror("data/bpe/valid.u16"); return 1; }
	uint16_t *tok = malloc((size_t)(warm + L + 1) * 2);
	if (fread(tok, 2, (size_t)(warm + L + 1), f) != (size_t)(warm + L + 1)) return 1;
	fclose(f);
	uint32_t *ids = malloc((size_t)(warm + L + 1) * 8);
	for (int t = 0; t < warm + L + 1; t++) ids[t] = tok[t];

	double *ring = malloc((long)CX_RING * CX_DB * CX_NBUS * 8);
	uint32_t *dummy = malloc(4 * 64);
	cx_in_init(dummy, 64, ring, 0);
	const double **bts = malloc((size_t)(warm + L) * sizeof *bts);
	for (int t = 0; t < warm + L; t++) bts[t] = ring + (long)(t % CX_RING) * CX_DB * CX_NBUS;

	/* warm-up: T from zero through WARMUP steps, no gradient */
	double *T0 = calloc(CX_NT, sizeof *T0);
	rstep *steps = malloc((size_t)(warm > L ? warm : L) * sizeof *steps);
	for (int t = 0; t < warm; t++) ref_forward(&P, T0, &steps[t], ids[t], bts[t]);
	double tn = 0;
	for (int i = 0; i < CX_NT; i++) tn += T0[i] * T0[i];

	rgrads_zero(&G);
	const double loss = ref_chunk(&P, &G, T0, ids + warm, bts + warm, L, steps);
	printf("train-ref: L %d, warm-up %d, |T0| %.4f, loss %.6f nats (%.4f bits/token), tokens %u %u %u ... -> %u\n", L, warm, sqrt(tn), loss,
	       loss / log(2.0), ids[warm], ids[warm + 1], ids[warm + 2], ids[warm + L]);

	uint64_t rs = 0x5EED5EED5EEDull;
	const double hstep = 1e-6;
	printf("%-5s %8s %13s %13s %13s\n", "group", "samples", "max |grad|", "max rel err", "worst fd/grad");
	for (int g = 0; g < 9; g++) {
		double gmax = 0, worst = 0, wfd = 0, wgr = 0;
		const double *gp = g == 7 ? G.cg : g == 8 ? G.E : G.prm + (g == 0 ? CX_O_PIN : g == 1 ? CX_O_WQB : g == 2 ? CX_O_GAIN : g == 3 ? CX_O_W : g == 4 ? CX_O_W1 : g == 5 ? CX_O_B1 : CX_O_WU);
		for (long i = 0; i < gsize[g]; i++) if (fabs(gp[i]) > gmax) gmax = fabs(gp[i]);
		for (int sidx = 0; sidx < samples; sidx++) {
			long idx;
			if (g == 8 && sidx < samples / 2) idx = (long)ids[warm + sidx % (L + 1)] * CX_DE + (long)(cx_rng(&rs) % CX_DE);   /* rows in use */
			else if (g == 3 && sidx < 4) idx = (long)(cx_rng(&rs) % CX_DX) * CX_NW + 192 + sidx % 2;                             /* a, b columns */
			else idx = (long)(cx_rng(&rs) % (uint64_t)gsize[g]);
			if (g == 3 && idx % CX_NW >= 194) idx -= idx % CX_NW;   /* padding columns carry no gradient; move to q */
			double *p = slot(&P, g, idx);
			const double keep = *p;
			*p = keep + hstep;
			const double lp = ref_chunk(&P, NULL, T0, ids + warm, bts + warm, L, steps);
			*p = keep - hstep;
			const double lm = ref_chunk(&P, NULL, T0, ids + warm, bts + warm, L, steps);
			*p = keep;
			const double fd = (lp - lm) / (2 * hstep), gr = gp[idx];
			const double den = fabs(fd) > fabs(gr) ? fabs(fd) : fabs(gr);
			const double rel = fabs(fd - gr) / (den > 1e-10 ? den : 1e-10);
			if (rel > worst) { worst = rel; wfd = fd; wgr = gr; }
		}
		printf("%-5s %8d %13.4e %13.3e %13.4e / %.4e\n", gname[g], samples, gmax, worst, wfd, wgr);
	}
	return 0;
}
