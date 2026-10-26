/*
 * isa-host (F64 port): the host makes the inputs and judges the card's F64 outputs against long double.
 *   isa-host gen DIR     writes DIR/{exp2,rcp,rsqrt,log2,hsum,hmax,sigmoid}.in, 2^18 doubles each
 *   isa-host check DIR   reads each .in with its .out (from card/isa.c) and prints the error statistics
 * ULP of a double reference r: 2^(e - 52) with e = floor(log2 |r|). Subnormal references accept 0 (flush to zero) or the exact
 * value. Sets: exp2 x uniform in [-120, 120], [-1, 1] and [-1100, 1100]; exp in [-40, 40], [-800, 800] and +-2^(20..1000);
 * rcp |x| log-uniform in [2^-100, 2^100] with random sign; rsqrt x log-uniform; log2 x log-uniform in [2^-100, 2^100] and
 * dense [0.5, 2]; sigmoid x uniform in [-40, 40]; hsum / hmax: groups of 8 uniform in [-1, 1].
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N (1 << 18)

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static double uni(void)
{
	rng ^= rng >> 12;
	rng ^= rng << 25;
	rng ^= rng >> 27;
	return (double)((rng * 0x2545F4914F6CDD1Dull) >> 11) / 9007199254740992.0;
}

static int put(const char *dir, const char *name, const double *v)
{
	char path[4096];
	snprintf(path, sizeof path, "%s/%s.in", dir, name);
	FILE *f = fopen(path, "wb");
	if (!f || fwrite(v, sizeof *v, N, f) != N || fclose(f)) { perror(path); return -1; }
	return 0;
}

static double *get(const char *dir, const char *name, const char *ext)
{
	char path[4096];
	snprintf(path, sizeof path, "%s/%s.%s", dir, name, ext);
	FILE *f = fopen(path, "rb");
	double *v = malloc(N * sizeof *v);
	if (!f || fread(v, sizeof *v, N, f) != N) { perror(path); exit(1); }
	fclose(f);
	return v;
}

static long double ulp_of(long double r)
{
	if (r == 0 || !isfinite((double)r)) return ldexpl(1, -1074);
	int k;
	frexpl(r, &k);
	const int e = k - 1 < -1022 ? -1022 : k - 1;
	return ldexpl(1, e - 52);
}

static int cmpd(const void *a, const void *b) { const double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

static void report(const char *name, const double *in, const double *out, long double (*ref)(long double), int lanes)
{
	double *ulps = malloc(N * sizeof *ulps);
	long n = 0, over1 = 0, disagree = 0, flushed = 0;
	double worst = 0, win = 0, wout = 0;
	long double wref = 0;
	for (long i = 0; i < N; i += lanes) {
		long double r;
		if (lanes == 8) {
			if (!strcmp(name, "hsum")) { r = 0; for (int l = 0; l < 8; l++) r += in[i + l]; }
			else { r = in[i]; for (int l = 1; l < 8; l++) if (in[i + l] > r) r = in[i + l]; }
			for (int l = 1; l < 8; l++) disagree += out[i + l] != out[i];
		} else r = ref(in[i]);
		if (!isfinite((double)r) || r == 0) { if (out[i] != (double)r) disagree++; continue; }
		if (fabsl(r) < 2.2250738585072014e-308) { if (out[i] != 0 && out[i] != (double)r) disagree++; flushed++; continue; }   /* subnormal result: 0 accepted (flush to zero) */
		const double u = (double)(fabsl((long double)out[i] - r) / ulp_of(r));
		ulps[n++] = u;
		if (u > 1) over1++;
		if (u > worst) { worst = u; win = in[i]; wout = out[i]; wref = r; }
	}
	qsort(ulps, (size_t)n, sizeof *ulps, cmpd);
	printf("%-8s n %7ld  max %.4f ULP  median %.4f  p99.9 %.4f  over 1 ULP %ld  disagreements %ld  subnormal %ld  (worst at x=%.17g: card %.17g ref %.17Lg)\n",
	       name, n, worst, n ? ulps[n / 2] : 0, n ? ulps[(long)(n * 0.999)] : 0, over1, disagree, flushed, win, wout, wref);
	free(ulps);
}

static long double r_exp2(long double x) { return exp2l(x); }
static long double r_rcp(long double x) { return 1.0L / x; }
static long double r_rsqrt(long double x) { return 1.0L / sqrtl(x); }
static long double r_log2(long double x) { return log2l(x); }
static long double r_sig(long double x) { return 1.0L / (1.0L + expl(-x)); }
static long double r_exp(long double x) { return expl(x); }

int main(int argc, char **argv)
{
	if (argc < 3) { fprintf(stderr, "usage: isa-host gen|check DIR\n"); return 2; }
	const char *dir = argv[2];
	if (!strcmp(argv[1], "gen")) {
		double *v = malloc(N * sizeof *v);
		for (long i = 0; i < N; i++) v[i] = i < N / 2 ? -120 + 240 * uni() : i < 3 * N / 4 ? -1 + 2 * uni() : -1100 + 2200 * uni();
		if (put(dir, "exp2", v)) return 1;
		for (long i = 0; i < N; i++) v[i] = (uni() < 0.5 ? -1 : 1) * exp2(-100 + 200 * uni());
		if (put(dir, "rcp", v)) return 1;
		for (long i = 0; i < N; i++) v[i] = exp2(-100 + 200 * uni());
		if (put(dir, "rsqrt", v)) return 1;
		for (long i = 0; i < N; i++) v[i] = i < N / 2 ? exp2(-100 + 200 * uni()) : 0.5 + 1.5 * uni();
		if (put(dir, "log2", v)) return 1;
		for (long i = 0; i < N; i++) v[i] = -40 + 80 * uni();
		if (put(dir, "sigmoid", v)) return 1;
		for (long i = 0; i < N; i++) v[i] = i < N / 2 ? -40 + 80 * uni() : i < N - 1024 ? -800 + 1600 * uni() : (i & 1 ? -1 : 1) * exp2(20 + 980 * uni());   /* wide: overflow, underflow, huge arguments */
		if (put(dir, "exp", v)) return 1;
		for (long i = 0; i < N; i++) v[i] = -1 + 2 * uni();
		if (put(dir, "hsum", v) || put(dir, "hmax", v)) return 1;
		printf("isa-host: %d doubles per set written to %s\n", N, dir);
		return 0;
	}
	report("exp2", get(dir, "exp2", "in"), get(dir, "exp2", "out"), r_exp2, 1);
	report("rcp", get(dir, "rcp", "in"), get(dir, "rcp", "out"), r_rcp, 1);
	report("rsqrt", get(dir, "rsqrt", "in"), get(dir, "rsqrt", "out"), r_rsqrt, 1);
	report("log2", get(dir, "log2", "in"), get(dir, "log2", "out"), r_log2, 1);
	{ const double *in = get(dir, "log2", "in"), *out = get(dir, "log2", "out"); double amax = 0; long n = 0; for (long i = 0; i < N; i++) if (in[i] >= 0.5 && in[i] <= 2) { const double d = fabs((double)(out[i] - log2l(in[i]))); if (d > amax) amax = d; n++; } printf("log2 on [0.5, 2]: n %ld, max absolute error %.3e (2^-52 = 2.2e-16)\n", n, amax); }
	report("sigmoid", get(dir, "sigmoid", "in"), get(dir, "sigmoid", "out"), r_sig, 1);
	report("exp", get(dir, "exp", "in"), get(dir, "exp", "out"), r_exp, 1);
	report("hsum", get(dir, "hsum", "in"), get(dir, "hsum", "out"), NULL, 8);
	report("hmax", get(dir, "hmax", "in"), get(dir, "hmax", "out"), NULL, 8);
	return 0;
}
