/*
 * vpu.h (F64 port): KNC F64 vector helpers on the S3 clang builtins (BuiltinsX86KNC.td). 8 lanes of double per 512-bit
 * vector. KNC has F64 add, sub, mul, FMA, max/min, rounding and conversions, but its reciprocal, rsqrt, exp2 and log2
 * instructions exist only in F32 (Intel 327364-001 pp. 190, 373, 577, 588). Here:
 *   rcp(x)   = F32 vrcp23ps estimate widened, then 3 Newton steps r = r (2 - x r)            (23 -> 46 -> 53 bits)
 *              CONTRACT: x must lie in the F32 normal range, 1.18e-38 <= |x| < 3.4e38; the seed is taken on the narrowed
 *              value, so a smaller |x| seeds with inf (and Newton makes NaN) and a larger one seeds with 0. Every caller
 *              in the model satisfies this (norms carry eps 1e-6, softmax sums are >= 1, Adam adds 1e-30). The two exact
 *              limits are handled: +-inf -> 0 (sigmoid of a very negative argument), 0 -> +inf.
 *   rsqrt(x) = F32 vrsqrt23ps estimate widened, then 3 Newton steps r = r (1.5 - x r^2 / 2); same range contract as rcp
 *   exp2(x)  = 2^n * 2^f, n = round(x), f in [-0.5, 0.5]: 2^f by a degree-13 Taylor series in f ln 2 (|f ln 2| <= 0.347,
 *              truncation below 5e-17), 2^n exact from the exponent field (cx_pow2i; subnormal results flush to 0),
 *              exp(x) = 2^(x log2e) with a split constant and an FMA argument reduction (cx_exp)
 *   log2(x)  = F32 vlog2ps estimate y0 widened, then one correction step: w = x 2^-y0 - 1 (|w| < 3e-5),
 *              y = y0 + (w - w^2/2 + w^3/3) / ln 2  (the cubic term keeps the step quadratic-plus; w^4/4 < 2e-19)
 * The F64 ISA gate (tools/isa-host.c) measures each against long double.
 */
#ifndef CX_VPU_H
#define CX_VPU_H

#include <immintrin.h>

typedef float v16f __attribute__((vector_size(64)));
typedef int v16i __attribute__((vector_size(64)));
typedef double v8d __attribute__((vector_size(64)));

#define CX_ALL ((unsigned short)0xffff)
#define CX_ALL8 ((unsigned char)0xff)
#define CX_FXP_8_24_RN 0x50

static inline __m512d cx_set1(double x) { return _mm512_set1_pd(x); }

/* conversions between the 8 doubles and the low 8 lanes of a 16-float vector */
static inline __m512 cx_pd2ps(__m512d x) { return (__m512)__builtin_ia32_knc_vcvtpd2ps_zz((v16f)_mm512_undefined_ps(), CX_ALL8, (v8d)x, 0); }
static inline __m512d cx_ps2pd(__m512 x) { return (__m512d)__builtin_ia32_knc_vcvtps2pd_zz((v8d)_mm512_undefined_pd(), CX_ALL8, (v16f)x, 0); }

static inline __m512 cx_rcp23(__m512 x) { return (__m512)__builtin_ia32_knc_vrcp23ps_zz((v16f)_mm512_undefined_ps(), CX_ALL, (v16f)x, 0); }
static inline __m512 cx_rsqrt23(__m512 x) { return (__m512)__builtin_ia32_knc_vrsqrt23ps_zz((v16f)_mm512_undefined_ps(), CX_ALL, (v16f)x, 0); }
static inline __m512 cx_log2ps(__m512 x) { return (__m512)__builtin_ia32_knc_vlog2ps_zz((v16f)_mm512_undefined_ps(), CX_ALL, (v16f)x, 0); }
static inline __m512 cx_exp2ps(__m512 x)
{
	const v16i q = __builtin_ia32_knc_vcvtfxpntps2dq_zzi((v16i)_mm512_undefined_epi32(), CX_ALL, (v16f)x, CX_FXP_8_24_RN, 0);
	return (__m512)__builtin_ia32_knc_vexp223ps_zz((v16f)_mm512_undefined_ps(), CX_ALL, (v16f)q, 0);
}

/* c - a b */
static inline __m512d cx_fnmadd(__m512d a, __m512d b, __m512d c)
{
	return (__m512d)__builtin_ia32_knc_vfnmadd231pd_zzz((v8d)c, CX_ALL8, (v8d)a, (v8d)b, 0);
}

/* lane helpers */
#define CX_BC(p) ((__m512d)__builtin_ia32_knc_vbroadcastsd_zm((v8d)_mm512_undefined_pd(), CX_ALL8, (p), 0))
static inline __m512d cx_blend(__m512d a, __mmask8 k, __m512d b) { return (__m512d)__builtin_ia32_knc_vmovapd_zz((v8d)a, k, (v8d)b, 0); }
static inline void cx_mask_store(double *p, __mmask8 k, __m512d v) { __builtin_ia32_knc_vmovapd_mz(k, p, (v8d)v, 0); }
static inline __m512d cx_max(__m512d a, __m512d b) { return (__m512d)__builtin_ia32_knc_vgmaxpd_zzz((v8d)_mm512_undefined_pd(), CX_ALL8, (v8d)a, (v8d)b, 0); }
#define CX_P128(v, imm) ((__m512d)__builtin_ia32_knc_vpermf32x4_zzi((v16f)_mm512_undefined_ps(), CX_ALL, (v16f)(__m512d)(v), (imm)))

static inline __m512d cx_rcp(__m512d x)
{
	__m512d r = cx_ps2pd(cx_rcp23(cx_pd2ps(x)));
	const __m512d two = cx_set1(2.0);
	for (int i = 0; i < 3; i++) r = _mm512_mul_pd(r, cx_fnmadd(x, r, two));
	/* the Newton step turns a zero seed (x = +-inf) into 0 * (2 - inf * 0) = NaN: +-inf -> 0 and 0 -> +inf explicitly */
	r = cx_blend(r, _mm512_cmp_pd_mask(cx_set1(3.4028234663852886e38), x, _CMP_LT_OS), _mm512_setzero_pd());
	r = cx_blend(r, _mm512_cmp_pd_mask(x, cx_set1(-3.4028234663852886e38), _CMP_LT_OS), _mm512_setzero_pd());
	r = cx_blend(r, _mm512_cmp_pd_mask(x, _mm512_setzero_pd(), _CMP_EQ_OQ), cx_set1(__builtin_inf()));
	return r;
}

static inline __m512d cx_rsqrt(__m512d x)
{
	__m512d r = cx_ps2pd(cx_rsqrt23(cx_pd2ps(x)));
	const __m512d half = cx_set1(0.5), three_half = cx_set1(1.5);
	const __m512d hx = _mm512_mul_pd(x, half);
	for (int i = 0; i < 3; i++) r = _mm512_mul_pd(r, cx_fnmadd(_mm512_mul_pd(hx, r), r, three_half));
	/* same guards: +inf -> 0, 0 -> +inf */
	r = cx_blend(r, _mm512_cmp_pd_mask(cx_set1(3.4028234663852886e38), x, _CMP_LT_OS), _mm512_setzero_pd());
	r = cx_blend(r, _mm512_cmp_pd_mask(x, _mm512_setzero_pd(), _CMP_EQ_OQ), cx_set1(__builtin_inf()));
	return r;
}

/* round to nearest integer, result a double (vrndfxpntpd, imm 0 = round to nearest even, no exponent adjustment) */
static inline __m512d cx_round(__m512d x) { return (__m512d)__builtin_ia32_knc_vrndfxpntpd_zzi((v8d)_mm512_undefined_pd(), CX_ALL8, (v8d)x, 0, 0); }


/* 2^n for an integer-valued n, exact for n in [-1022, 1023] (clamped), built in the double's exponent field.
 * KNC has no 64-bit integer add or shift, so: k = n + 1023; the double 2^52 + k 2^20 (exact) has the bit pattern
 * hi 0x43300000, lo k << 20; a 32-bit CDAB swizzle swaps the two words of each lane (hi = k << 20); a masked vmovdqa32
 * zeroes the even (low) words; the result reinterpreted as a double is 2^n. */
static inline __m512d cx_pow2i(__m512d n)
{
	n = (__m512d)__builtin_ia32_knc_vgmaxpd_zzz((v8d)_mm512_undefined_pd(), CX_ALL8, (v8d)n, (v8d)cx_set1(-1022.0), 0);
	n = (__m512d)__builtin_ia32_knc_vgminpd_zzz((v8d)_mm512_undefined_pd(), CX_ALL8, (v8d)n, (v8d)cx_set1(1023.0), 0);
	const __m512d t = _mm512_fmadd_pd(_mm512_add_pd(n, cx_set1(1023.0)), cx_set1(1048576.0), cx_set1(4503599627370496.0));
	const __m512i sw = _mm512_swizzle_epi32(_mm512_castpd_si512(t), _MM_SWIZ_REG_CDAB);
	return _mm512_castsi512_pd((__m512i)__builtin_ia32_knc_vmovdqa32_zz((v16i)_mm512_setzero_epi32(), (unsigned short)0xAAAA, (v16i)sw, 0));
}

/* 2^f for f in [-0.5, 0.5]: e^(f ln 2) by a degree-13 Taylor series in Horner form (u^14 / 14! < 5e-17 at |u| <= 0.347) */
static inline __m512d cx_exp2_frac(__m512d f)
{
	const __m512d u = _mm512_mul_pd(f, cx_set1(0.6931471805599453));
	__m512d p = cx_set1(1.0 / 6227020800.0);
	p = _mm512_fmadd_pd(p, u, cx_set1(1.0 / 479001600.0));
	p = _mm512_fmadd_pd(p, u, cx_set1(1.0 / 39916800.0));
	p = _mm512_fmadd_pd(p, u, cx_set1(1.0 / 3628800.0));
	p = _mm512_fmadd_pd(p, u, cx_set1(1.0 / 362880.0));
	p = _mm512_fmadd_pd(p, u, cx_set1(1.0 / 40320.0));
	p = _mm512_fmadd_pd(p, u, cx_set1(1.0 / 5040.0));
	p = _mm512_fmadd_pd(p, u, cx_set1(1.0 / 720.0));
	p = _mm512_fmadd_pd(p, u, cx_set1(1.0 / 120.0));
	p = _mm512_fmadd_pd(p, u, cx_set1(1.0 / 24.0));
	p = _mm512_fmadd_pd(p, u, cx_set1(1.0 / 6.0));
	p = _mm512_fmadd_pd(p, u, cx_set1(0.5));
	p = _mm512_fmadd_pd(p, u, cx_set1(1.0));
	return _mm512_fmadd_pd(p, u, cx_set1(1.0));
}

/* 2^x: results below 2^-1022 (subnormal) flush to 0; x >= 1024 gives +inf */
static inline __m512d cx_exp2(__m512d x)
{
	const __m512d n = cx_round(x);
	__m512d r = _mm512_mul_pd(cx_exp2_frac(_mm512_sub_pd(x, n)), cx_pow2i(n));
	r = cx_blend(r, _mm512_cmp_pd_mask(cx_set1(1023.0), n, _CMP_LT_OS), _mm512_add_pd(r, r));   /* n >= 1024: one more doubling, inf only on true overflow */
	r = cx_blend(r, _mm512_cmp_pd_mask(cx_set1(1024.0), n, _CMP_LT_OS), cx_set1(__builtin_inf()));   /* n >= 1025: overflow */
	return cx_blend(r, _mm512_cmp_pd_mask(x, cx_set1(-1022.0), _CMP_LT_OS), _mm512_setzero_pd());
}

/* e^x = 2^(x log2 e) with log2 e = Lhi + Llo: n = round(x Lhi), f = fma(x, Lhi, -n) + x Llo (the FMA rounds once, on a
 * small quantity; a plain product rounding of x log2 e costs about |x| 2^-53 relative, 72 ULP at |x| = 40). The argument
 * is clamped to [-1000, 1000] first: for huge |x| the FMA residual is the product's own rounding error (1e284 at 1e300),
 * the polynomial would overflow and inf * 0 = NaN. Results below 2^-1022 flush to 0 (x < -708.3964); x > 709.78 gives +inf. */
static inline __m512d cx_exp(__m512d x)
{
	x = (__m512d)__builtin_ia32_knc_vgmaxpd_zzz((v8d)_mm512_undefined_pd(), CX_ALL8, (v8d)x, (v8d)cx_set1(-1000.0), 0);
	x = (__m512d)__builtin_ia32_knc_vgminpd_zzz((v8d)_mm512_undefined_pd(), CX_ALL8, (v8d)x, (v8d)cx_set1(1000.0), 0);
	const __m512d Lhi = cx_set1(1.4426950408889634), Llo = cx_set1(2.0355273740931033e-17);
	const __m512d n = cx_round(_mm512_mul_pd(x, Lhi));
	const __m512d f = _mm512_fmadd_pd(x, Llo, _mm512_fmadd_pd(x, Lhi, _mm512_sub_pd(_mm512_setzero_pd(), n)));
	__m512d r = _mm512_mul_pd(cx_exp2_frac(f), cx_pow2i(n));
	r = cx_blend(r, _mm512_cmp_pd_mask(cx_set1(1023.0), n, _CMP_LT_OS), _mm512_add_pd(r, r));
	r = cx_blend(r, _mm512_cmp_pd_mask(cx_set1(1024.0), n, _CMP_LT_OS), cx_set1(__builtin_inf()));   /* n >= 1025: overflow */
	return cx_blend(r, _mm512_cmp_pd_mask(x, cx_set1(-708.3964185322641), _CMP_LT_OS), _mm512_setzero_pd());
}

static inline __m512d cx_log2(__m512d x)
{
	const __m512d y0 = cx_ps2pd(cx_log2ps(cx_pd2ps(x)));
	const __m512d w = _mm512_sub_pd(_mm512_mul_pd(x, cx_exp2(_mm512_sub_pd(_mm512_setzero_pd(), y0))), cx_set1(1.0));   /* x 2^-y0 - 1 */
	const __m512d w2 = _mm512_mul_pd(w, w);
	const __m512d c = _mm512_fmadd_pd(_mm512_mul_pd(w2, w), cx_set1(1.0 / 3.0), cx_fnmadd(w2, cx_set1(0.5), w));   /* w - w^2/2 + w^3/3 */
	return _mm512_fmadd_pd(c, cx_set1(1.4426950408889634), y0);
}


/* 1 / (1 + e^-x) */
static inline __m512d cx_sigmoid(__m512d x)
{
	const __m512d e = cx_exp(_mm512_sub_pd(_mm512_setzero_pd(), x));
	return cx_rcp(_mm512_add_pd(e, cx_set1(1.0)));
}


/* horizontal sum and max: 64-bit swizzles act within 256-bit halves (cdab, badc), then the halves are swapped */
static inline __m512d cx_hsum(__m512d v)
{
	v = _mm512_add_pd(v, _mm512_swizzle_pd(v, _MM_SWIZ_REG_CDAB));
	v = _mm512_add_pd(v, _mm512_swizzle_pd(v, _MM_SWIZ_REG_BADC));
	v = _mm512_add_pd(v, CX_P128(v, 0x4E));
	return v;
}

static inline __m512d cx_hmax(__m512d v)
{
	v = cx_max(v, _mm512_swizzle_pd(v, _MM_SWIZ_REG_CDAB));
	v = cx_max(v, _mm512_swizzle_pd(v, _MM_SWIZ_REG_BADC));
	v = cx_max(v, CX_P128(v, 0x4E));
	return v;
}

#endif
