/*
 * col-ref: Stage 1, host side. usage: col-ref DIR STEPS THREADS DUMP_THREAD...
 *
 * Regenerates the same F32 weights and inputs as the card (lib/cx-col.h), runs the column forward in double for
 * STEPS steps, and compares with the card's dumps DIR/col-nN-tK.f32 (u, bc, g per step) and -T.f32 (final fast
 * weights). Prints, per output, the largest absolute error and the largest error relative to that vector's largest
 * reference magnitude, over all steps. A wiring fault shows as O(1); F32 FMAs plus the 1-ULP transcendental
 * instructions should stay below ~1e-4.
 */
#include "../lib/cx-col.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double *E;
static int ringn = CX_RING;

typedef struct { double abs, rel; } err;

static void worst(err *e, const double *card, const double *ref, int n)
{
	double scale = 0, a = 0;
	for (int i = 0; i < n; i++) if (fabs(ref[i]) > scale) scale = fabs(ref[i]);
	for (int i = 0; i < n; i++) { const double d = fabs(card[i] - ref[i]); if (d > a) a = d; }
	if (a > e->abs) e->abs = a;
	if (scale > 0 && a / scale > e->rel) e->rel = a / scale;
}

static double sigm(double x) { return 1.0 / (1.0 + exp(-x)); }

static void gemv(const double *W, const double *x, int K, int N, double *y)
{
	for (int n = 0; n < N; n++) y[n] = 0;
	for (int k = 0; k < K; k++)
		for (int n = 0; n < N; n++) y[n] += x[k] * W[(long)k * N + n];
}

static int check(const char *dir, int steps, int nthr, int k)
{
	double *prm = malloc(CX_NPARAM * 8), *ring = malloc((long)CX_RING * CX_DB * CX_NBUS * 8);
	uint32_t *ids = malloc((size_t)steps * 4);
	cx_col_init(prm, k);
	cx_in_init(ids, steps, ring, k);
	double T[CX_DS][CX_DS];
	memset(T, 0, sizeof T);

	char path[4096];
	snprintf(path, sizeof path, "%s/col-n%d-t%d.f32", dir, nthr, k);
	FILE *f = fopen(path, "rb");
	double *dump = malloc(((size_t)steps * CX_DUMP + CX_NACT) * 8);
	if (!f || fread(dump, 8, (size_t)steps * CX_DUMP + CX_NACT, f) != (size_t)steps * CX_DUMP + CX_NACT) { perror(path); return 1; }
	const double *act0 = dump + (long)steps * CX_DUMP;
	fclose(f);
	snprintf(path, sizeof path, "%s/col-n%d-t%d-T.f32", dir, nthr, k);
	f = fopen(path, "rb");
	static double Tc[CX_NT];
	if (!f || fread(Tc, 8, CX_NT, f) != CX_NT) { perror(path); return 1; }
	fclose(f);

	err eu = { 0, 0 }, ebc = { 0, 0 }, eg = { 0, 0 }, eT = { 0, 0 };
	double gmin = 1, gmax = 0, umax = 0, amax = 0, bmax = 0, alphamax = 0;
	for (int t = 0; t < steps; t++) {
		const double *Bt = ring + (long)(t % ringn) * CX_DB * CX_NBUS;
		double e[CX_DE], p[CX_DP], qb[CX_DB], s[CX_NBUS], alpha[CX_NBUS], r[CX_DB], xin[CX_DX], min[CX_DM], y[CX_NW];
		double kv[CX_DS], Sk[CX_DS], Sq[CX_DS], d[CX_DS], h[CX_DH], z[CX_DH], u[CX_DU], bc[CX_DB];
		for (int i = 0; i < CX_DE; i++) e[i] = E[(long)ids[t] * CX_DE + i];
		gemv(prm + CX_O_PIN, e, CX_DE, CX_DP, p);
		gemv(prm + CX_O_WQB, p, CX_DP, CX_DB, qb);
		gemv(Bt, qb, CX_DB, CX_NBUS, s);
		double m = -INFINITY, sum = 0;
		for (int j = 0; j < CX_NCOL; j++) if (s[j] > m) m = s[j];
		for (int j = 0; j < CX_NBUS; j++) { alpha[j] = j < CX_NCOL ? exp(s[j] - m) : 0; sum += alpha[j]; }
		for (int j = 0; j < CX_NBUS; j++) { alpha[j] /= sum; if (alpha[j] > alphamax) alphamax = alpha[j]; }
		for (int dd = 0; dd < CX_DB; dd++) { r[dd] = 0; for (int j = 0; j < CX_NBUS; j++) r[dd] += alpha[j] * Bt[dd * CX_NBUS + j]; }
		double ss = 0;
		for (int i = 0; i < CX_DX; i++) { xin[i] = i < CX_DP ? p[i] : r[i - CX_DP]; ss += xin[i] * xin[i]; }
		const double xs = 1.0 / sqrt(ss / CX_DX + (double)CX_EPS);
		for (int i = 0; i < CX_DX; i++) min[CX_DS + i] = xin[i] * xs * prm[CX_O_GAIN + i];
		gemv(prm + CX_O_W, min + CX_DS, CX_DX, CX_NW, y);
		double kn = 0;
		for (int i = 0; i < CX_DS; i++) kn += y[64 + i] * y[64 + i];
		const double ks = 1.0 / sqrt(kn + (double)CX_EPS);
		for (int i = 0; i < CX_DS; i++) kv[i] = y[64 + i] * ks;
		const double a = sigm(y[192]), b = sigm(y[193]);
		if (a > amax) amax = a;
		if (b > bmax) bmax = b;
		for (int i = 0; i < CX_DS; i++) { Sk[i] = Sq[i] = 0; }
		for (int j = 0; j < CX_DS; j++)
			for (int i = 0; i < CX_DS; i++) { Sk[i] += kv[j] * T[j][i]; Sq[i] += y[j] * T[j][i]; }
		double kq = 0;
		for (int i = 0; i < CX_DS; i++) { d[i] = b * (y[128 + i] - a * Sk[i]); kq += kv[i] * y[i]; }
		for (int i = 0; i < CX_DS; i++) min[i] = a * Sq[i] + kq * d[i];
		for (int j = 0; j < CX_DS; j++)
			for (int i = 0; i < CX_DS; i++) T[j][i] = a * T[j][i] + kv[j] * d[i];
		gemv(prm + CX_O_W1, min, CX_DM, CX_DH, h);
		for (int i = 0; i < CX_DH; i++) { h[i] += prm[CX_O_B1 + i]; z[i] = h[i] * sigm(h[i]); }
		gemv(prm + CX_O_WU, z, CX_DH, CX_DU, u);
		double gl = prm[CX_O_BG];
		for (int i = 0; i < CX_DH; i++) gl += z[i] * prm[CX_O_WG + i];
		const double g = sigm(gl);
		gemv(prm + CX_O_WB, z, CX_DH, CX_DB, bc);
		for (int i = 0; i < CX_DU; i++) if (fabs(u[i]) > umax) umax = fabs(u[i]);
		if (g < gmin) gmin = g;
		if (g > gmax) gmax = g;

		const double *o = dump + (long)t * CX_DUMP;
		double gv[16];
		for (int i = 0; i < 16; i++) gv[i] = g;
		worst(&eu, o, u, CX_DU);
		worst(&ebc, o + 128, bc, CX_DB);
		worst(&eg, o + 144, gv, 8);
		if (t == 0) {
			err ep = { 0, 0 }, eqb = { 0, 0 }, eal = { 0, 0 }, ex = { 0, 0 }, ey = { 0, 0 }, ek = { 0, 0 }, eo = { 0, 0 }, eh = { 0, 0 }, ez = { 0, 0 };
			worst(&ep, act0 + CX_A_P, p, CX_DP); worst(&eqb, act0 + CX_A_QB, qb, CX_DB); worst(&eal, act0 + CX_A_ALPHA, alpha, CX_NBUS);
			worst(&ex, act0 + CX_A_MIN + CX_DS, min + CX_DS, CX_DX); worst(&ey, act0 + CX_A_Y, y, 194); worst(&ek, act0 + CX_A_K, kv, CX_DS);
			worst(&eo, act0 + CX_A_MIN, min, CX_DS); worst(&eh, act0 + CX_A_H, h, CX_DH); worst(&ez, act0 + CX_A_Z, z, CX_DH);
			printf("  step 0 intermediates, max rel err: p %.2e qb %.2e alpha %.2e x %.2e y %.2e k %.2e o %.2e h %.2e z %.2e\n", ep.rel, eqb.rel, eal.rel, ex.rel, ey.rel, ek.rel, eo.rel, eh.rel, ez.rel);
		}
	}
	double Tr[CX_NT];
	for (int j = 0; j < CX_DS; j++) for (int i = 0; i < CX_DS; i++) Tr[j * CX_DS + i] = T[j][i];
	worst(&eT, Tc, Tr, CX_NT);

	printf("thread %3d, %d steps: max abs err / max err rel to the vector's max |ref|\n", k, steps);
	printf("  u (128)   %.3e  %.3e\n", eu.abs, eu.rel);
	printf("  bc (16)   %.3e  %.3e\n", ebc.abs, ebc.rel);
	printf("  g         %.3e  %.3e\n", eg.abs, eg.rel);
	printf("  T final   %.3e  %.3e\n", eT.abs, eT.rel);
	printf("  reference ranges: |u| max %.4f, g %.4f..%.4f, a max %.4f, b max %.4f, alpha max %.4f\n", umax, gmin, gmax, amax, bmax, alphamax);
	free(prm); free(ring); free(ids); free(dump);
	return 0;
}

int main(int argc, char **argv)
{
	if (argc < 5) { fprintf(stderr, "usage: col-ref DIR STEPS THREADS DUMP_THREAD...\n"); return 2; }
	const int steps = atoi(argv[2]), nthr = atoi(argv[3]);
	if (getenv("CX_RINGN")) ringn = atoi(getenv("CX_RINGN"));
	if (ringn < 1 || ringn > CX_RING) ringn = CX_RING;
	E = malloc((long)CX_VOCAB * CX_DE * 8);
	cx_emb_init(E);
	int rc = 0;
	for (int i = 4; i < argc; i++) rc |= check(argv[1], steps, nthr, atoi(argv[i]));
	return rc;
}
