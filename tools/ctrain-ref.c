/*
 * ctrain-ref: Stage 4a (host). Finite-difference check of the cluster reference (tools/cref.h) with an external bus
 * (stop-gradient exact), then the size of the gradient path the live bus drops. usage: ctrain-ref [N] [L] [WARM] [SAMPLES]
 */
#include "cref.h"

#include <stdio.h>

static const char *gname[] = { "P_in", "Wqb", "gain", "W", "W1", "b1", "Wu", "wg", "bg", "Wb", "cg", "E" };
static const long goff[] = { CX_O_PIN, CX_O_WQB, CX_O_GAIN, CX_O_W, CX_O_W1, CX_O_B1, CX_O_WU, CX_O_WG, CX_O_BG, CX_O_WB, 0, 0 };
static const long gsize[] = { CX_DE * CX_DP, CX_DP * CX_DB, CX_DX, CX_DX * CX_NW, CX_DM * CX_DH, CX_DH, CX_DH * CX_DU, CX_DH, 1, CX_DH * CX_DB, CX_DU, (long)CX_VOCAB * CX_DE };

int main(int argc, char **argv)
{
	const int N = argc > 1 ? atoi(argv[1]) : 4, L = argc > 2 ? atoi(argv[2]) : 6, warm = argc > 3 ? atoi(argv[3]) : 6, samples = argc > 4 ? atoi(argv[4]) : 12;
	cref C;
	cref_alloc(&C, N, L > warm ? L : warm);
	cref_init(&C);
	C.L = L > warm ? L : warm;
	cgrads G;
	cgrads_alloc(&G, N);

	FILE *f = fopen("data/bpe/valid.u16", "rb");
	if (!f) { perror("data/bpe/valid.u16"); return 1; }
	uint16_t *tok = malloc((size_t)(warm + L + 1) * 2);
	if (fread(tok, 2, (size_t)(warm + L + 1), f) != (size_t)(warm + L + 1)) return 1;
	fclose(f);
	uint32_t *ids = malloc((size_t)(warm + L + 1) * 8);
	for (int t = 0; t < warm + L + 1; t++) ids[t] = tok[t];
	double *ring = malloc((long)CX_RING * CX_DB * CX_NBUS * 8);
	uint32_t dummy[CX_RING];
	cx_in_init(dummy, CX_RING, ring, 0);
	const double **ext = malloc((size_t)(warm + L) * sizeof *ext);
	for (int t = 0; t < warm + L; t++) ext[t] = ring + (long)(t % CX_RING) * CX_DB * CX_NBUS;

	/* warm-up with the live bus (state and bus carried), no gradient */
	double *T0 = calloc((size_t)N * CX_NT, sizeof *T0);
	double *bus0 = calloc(CX_DB * CX_NBUS, sizeof *bus0);
	C.bus_mode = CREF_BUS_LIVE;
	for (int t = 0; t < warm; t++) cref_step(&C, t, ids[t]);
	memcpy(T0, C.T, (size_t)N * CX_NT * sizeof *T0);
	memcpy(bus0, C.bus + (long)warm * CX_DB * CX_NBUS, CX_DB * CX_NBUS * sizeof *bus0);   /* the bus the last warm-up step wrote */

	/* A: external bus (stop-gradient exact) */
	C.bus_mode = CREF_BUS_EXT;
	C.ext = ext + warm;
	cgrads_zero(&G, N);
	const double loss = cref_chunk(&C, &G, T0, bus0, ids + warm, L);
	printf("ctrain-ref: N %d, L %d, warm %d: loss %.6f nats (%.4f bits/token), external bus\n", N, L, warm, loss, loss / log(2.0));
	uint64_t rs = 0x5EED5EED5EEDull;
	const double h = 1e-6;
	printf("%-5s %4s %8s %13s %13s %13s\n", "group", "col", "samples", "max |grad|", "max rel err", "worst fd/grad");
	for (int g = 0; g < 12; g++) {
		const int cols = g >= 10 ? 1 : 2;
		for (int ci = 0; ci < cols; ci++) {
			const int c = ci ? N - 1 : 0;
			double *base = g == 10 ? C.cg : g == 11 ? C.E : C.prm + (long)c * CX_NPARAM + goff[g];
			const double *gb = g == 10 ? G.cg : g == 11 ? G.E : G.prm + (long)c * CX_NPARAM + goff[g];
			double gmax = 0, worst = 0, wfd = 0, wgr = 0;
			for (long i = 0; i < gsize[g]; i++) if (fabs(gb[i]) > gmax) gmax = fabs(gb[i]);
			const int ns = gsize[g] < samples ? (int)gsize[g] : samples;
			for (int si = 0; si < ns; si++) {
				long idx;
				if (g == 11 && si < ns / 2) idx = (long)ids[warm + si % (L + 1)] * CX_DE + (long)(cx_rng(&rs) % CX_DE);
				else if (g == 3 && si < 2) idx = (long)(cx_rng(&rs) % CX_DX) * CX_NW + 192 + si;
				else idx = gsize[g] == 1 ? 0 : (long)(cx_rng(&rs) % (uint64_t)gsize[g]);
				if (g == 3 && idx % CX_NW >= 194) idx -= idx % CX_NW;
				const double keep = base[idx];
				base[idx] = keep + h;
				const double lp = cref_chunk(&C, NULL, T0, bus0, ids + warm, L);
				base[idx] = keep - h;
				const double lm = cref_chunk(&C, NULL, T0, bus0, ids + warm, L);
				base[idx] = keep;
				const double fd = (lp - lm) / (2 * h), gr = gb[idx];
				const double den = fabs(fd) > fabs(gr) ? fabs(fd) : fabs(gr);
				const double rel = fabs(fd - gr) / (den > 1e-10 ? den : 1e-10);
				if (rel > worst) { worst = rel; wfd = fd; wgr = gr; }
			}
			printf("%-5s %4d %8d %13.4e %13.3e %13.4e / %.4e\n", gname[g], c, ns, gmax, worst, wfd, wgr);
		}
	}

	/* B: the live bus: how large is the gradient path the stop-gradient drops? FD (true gradient) vs analytic */
	C.bus_mode = CREF_BUS_LIVE;
	cgrads_zero(&G, N);
	const double lossB = cref_chunk(&C, &G, T0, bus0, ids + warm, L);
	printf("live bus: loss %.6f nats; dropped-path check on column 0 (FD = true gradient incl. the bus path, grad = analytic without it):\n", lossB);
	const int gl[] = { 9, 6, 3, 0 };
	for (int gi = 0; gi < 4; gi++) {
		const int g = gl[gi];
		double *base = C.prm + goff[g];
		const double *gb = G.prm + goff[g];
		double worst_abs = 0, wfd = 0, wgr = 0, gmax = 0;
		for (long i = 0; i < gsize[g]; i++) if (fabs(gb[i]) > gmax) gmax = fabs(gb[i]);
		for (int si = 0; si < 8; si++) {
			const long idx = (long)(cx_rng(&rs) % (uint64_t)gsize[g]);
			const double keep = base[idx];
			base[idx] = keep + h;
			const double lp = cref_chunk(&C, NULL, T0, bus0, ids + warm, L);
			base[idx] = keep - h;
			const double lm = cref_chunk(&C, NULL, T0, bus0, ids + warm, L);
			base[idx] = keep;
			const double fd = (lp - lm) / (2 * h), gr = gb[idx];
			if (fabs(fd - gr) > worst_abs) { worst_abs = fabs(fd - gr); wfd = fd; wgr = gr; }
		}
		printf("  %-5s max |analytic| %.3e  worst |fd - grad| %.3e  (fd %.4e, grad %.4e)\n", gname[g], gmax, worst_abs, wfd, wgr);
	}
	return 0;
}
