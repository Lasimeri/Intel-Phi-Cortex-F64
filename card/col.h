/*
 * col.h (F64 port): one column's forward step on the 7120A vector unit in double, 8 lanes per 512-bit vector. Header
 * only, single thread. Every arithmetic instruction is a VPU instruction (kncintrin.h pd forms plus the raw KNC
 * builtins in vpu.h). Shapes and layout: lib/cx-col.h (the F32 layout in elements, now doubles). W is 208 wide =
 * 26 vectors, more than the register file allows as accumulators, so it is done in two 13-vector passes with a
 * leading dimension.
 */
#ifndef CX_COL_KERNEL_H
#define CX_COL_KERNEL_H

#include "vpu.h"
#include "../lib/cx-col.h"

#ifndef CX_PF
#define CX_PF 1
#endif

static inline void cx_pf0(const void *p) { __asm__ volatile("vprefetch0 %0" : : "m"(*(const char *)p)); }

/* y[8 NV] = x[K]^T W[K][..], W rows ld doubles apart (ld = 8 NV when contiguous): per k one broadcast and NV FMAs */
#define CX_GEMV(NV)                                                                                                  \
	static inline void cx_gemv##NV(const double *W, long ld, const double *x, int K, double *y)                      \
	{                                                                                                                \
		__m512d a[NV];                                                                                               \
		_Pragma("unroll") for (int v = 0; v < NV; v++) a[v] = _mm512_setzero_pd();                                   \
		for (int k = 0; k < K; k++) {                                                                                \
			const double *w = W + (long)k * ld;                                                                      \
			if (CX_PF) { _Pragma("unroll") for (int v = 0; v < NV; v++) cx_pf0(w + 2 * ld + 8 * v); }                \
			const __m512d xb = CX_BC(x + k);                                                                         \
			_Pragma("unroll") for (int v = 0; v < NV; v++) a[v] = _mm512_fmadd_pd(xb, _mm512_load_pd(w + 8 * v), a[v]); \
		}                                                                                                            \
		_Pragma("unroll") for (int v = 0; v < NV; v++) _mm512_store_pd(y + 8 * v, a[v]);                             \
	}
CX_GEMV(2)
CX_GEMV(8)
CX_GEMV(13)
CX_GEMV(16)
#define CX_NVH (CX_DH / 8)

typedef struct {
	double *prm;
	double *T;
	double *act;
} cx_col;

/* one token step. e: this token's embedding row (128). Bt: this step's bus input, [16][128]. Leaves u at act + CX_A_U,
 * g (all lanes) at CX_A_G, bc at CX_A_BC, and the values the backward needs at CX_A_R .. CX_A_SG. */
static inline void cx_col_step(cx_col *c, const double *e, const double *Bt)
{
	double *const A = c->act;
	const double *const P = c->prm;
	double *const T = c->T;
	const __m512d zero = _mm512_setzero_pd(), l2e = cx_set1(1.4426950408889634), one = cx_set1(1.0);

	/* p = e^T P_in; qb = p^T Wqb; scores s = qb^T Bt */
	cx_gemv8(P + CX_O_PIN, CX_DP, e, CX_DE, A + CX_A_P);
	cx_gemv2(P + CX_O_WQB, CX_DB, A + CX_A_P, CX_DP, A + CX_A_QB);
	cx_gemv16(Bt, CX_NBUS, A + CX_A_QB, CX_DB, A + CX_A_S);

	/* softmax over the 120 real slots = vectors 0..14; vector 15 is padding */
	__m512d s[16], m;
	for (int v = 0; v < 15; v++) s[v] = _mm512_load_pd(A + CX_A_S + 8 * v);
	m = s[0];
	for (int v = 1; v < 15; v++) m = cx_max(m, s[v]);
	m = cx_hmax(m);
	__m512d sum = zero;
	for (int v = 0; v < 15; v++) { s[v] = cx_exp(_mm512_sub_pd(s[v], m)); sum = _mm512_add_pd(sum, s[v]); }
	const __m512d inv = cx_rcp(cx_hsum(sum));
	for (int v = 0; v < 15; v++) _mm512_store_pd(A + CX_A_ALPHA + 8 * v, _mm512_mul_pd(s[v], inv));
	_mm512_store_pd(A + CX_A_ALPHA + 120, zero);

	/* r_d = alpha . Bt[d]: one lane per d, two vectors of 8 */
	__m512d r[2] = { zero, zero };
	for (int d = 0; d < CX_DB; d++) {
		const double *row = Bt + d * CX_NBUS;
		__m512d acc = _mm512_mul_pd(_mm512_load_pd(A + CX_A_ALPHA), _mm512_load_pd(row));
		for (int v = 1; v < 15; v++) acc = _mm512_fmadd_pd(_mm512_load_pd(A + CX_A_ALPHA + 8 * v), _mm512_load_pd(row + 8 * v), acc);
		r[d >> 3] = cx_blend(r[d >> 3], (__mmask8)(1u << (d & 7)), cx_hsum(acc));
	}
	_mm512_store_pd(A + CX_A_R, r[0]);
	_mm512_store_pd(A + CX_A_R + 8, r[1]);

	/* x = RMSNorm([p ; r]) * gain, stored as the tail of [o ; x] */
	__m512d xin[10];
	for (int v = 0; v < 8; v++) xin[v] = _mm512_load_pd(A + CX_A_P + 8 * v);
	xin[8] = r[0];
	xin[9] = r[1];
	__m512d ss = _mm512_mul_pd(xin[0], xin[0]);
	for (int v = 1; v < 10; v++) ss = _mm512_fmadd_pd(xin[v], xin[v], ss);
	ss = cx_hsum(ss);
	const __m512d xs = cx_rsqrt(_mm512_fmadd_pd(ss, cx_set1(1.0 / CX_DX), cx_set1(CX_EPS)));
	_mm512_store_pd(A + CX_A_XS, xs);
	double *const x = A + CX_A_MIN + CX_DS;
	for (int v = 0; v < 10; v++) _mm512_store_pd(x + 8 * v, _mm512_mul_pd(_mm512_mul_pd(xin[v], xs), _mm512_load_pd(P + CX_O_GAIN + 8 * v)));

	/* y = x^T W in two passes of 104 columns */
	double *const y = A + CX_A_Y;
	cx_gemv13(P + CX_O_W, CX_NW, x, CX_DX, y);
	cx_gemv13(P + CX_O_W + 104, CX_NW, x, CX_DX, y + 104);

	/* k = k / |k| */
	__m512d kk = _mm512_load_pd(y + 64), kn = _mm512_mul_pd(kk, kk);
	for (int v = 1; v < 8; v++) { kk = _mm512_load_pd(y + 64 + 8 * v); kn = _mm512_fmadd_pd(kk, kk, kn); }
	const __m512d ks = cx_rsqrt(_mm512_add_pd(cx_hsum(kn), cx_set1(CX_EPS)));
	_mm512_store_pd(A + CX_A_KS, ks);
	for (int v = 0; v < 8; v++) _mm512_store_pd(A + CX_A_K + 8 * v, _mm512_mul_pd(_mm512_load_pd(y + 64 + 8 * v), ks));

	const __m512d a = cx_sigmoid(CX_BC(y + 192)), b = cx_sigmoid(CX_BC(y + 193));
	_mm512_store_pd(A + CX_A_AV, a);
	_mm512_store_pd(A + CX_A_BV, b);

	/* pass 1 over T: Sk = S k and Sq = S q together */
	__m512d Sk[8], Sq[8];
	for (int v = 0; v < 8; v++) Sk[v] = Sq[v] = zero;
	for (int j = 0; j < CX_DS; j++) {
		const double *row = T + j * CX_DS;
		if (CX_PF) for (int v = 0; v < 8; v++) cx_pf0(row + 2 * CX_DS + 8 * v);
		const __m512d kj = CX_BC(A + CX_A_K + j), qj = CX_BC(y + j);
		for (int v = 0; v < 8; v++) {
			const __m512d t = _mm512_load_pd(row + 8 * v);
			Sk[v] = _mm512_fmadd_pd(kj, t, Sk[v]);
			Sq[v] = _mm512_fmadd_pd(qj, t, Sq[v]);
		}
	}
	/* d = b v - (a b) Sk;  kq = k . q;  o = a Sq + kq d */
	const __m512d ab = _mm512_mul_pd(a, b);
	__m512d d[8], kq = zero;
	for (int v = 0; v < 8; v++) {
		d[v] = cx_fnmadd(ab, Sk[v], _mm512_mul_pd(b, _mm512_load_pd(y + 128 + 8 * v)));
		kq = _mm512_fmadd_pd(_mm512_load_pd(A + CX_A_K + 8 * v), _mm512_load_pd(y + 8 * v), kq);
	}
	kq = cx_hsum(kq);
	_mm512_store_pd(A + CX_A_KQ, kq);
	for (int v = 0; v < 8; v++) {
		_mm512_store_pd(A + CX_A_SK + 8 * v, Sk[v]);
		_mm512_store_pd(A + CX_A_SQ + 8 * v, Sq[v]);
		_mm512_store_pd(A + CX_A_D + 8 * v, d[v]);
		_mm512_store_pd(A + CX_A_MIN + 8 * v, _mm512_fmadd_pd(kq, d[v], _mm512_mul_pd(a, Sq[v])));
	}

	/* pass 2 over T: T[j] = a T[j] + k_j d */
	for (int j = 0; j < CX_DS; j++) {
		double *row = T + j * CX_DS;
		const __m512d kj = CX_BC(A + CX_A_K + j);
		for (int v = 0; v < 8; v++) _mm512_store_pd(row + 8 * v, _mm512_fmadd_pd(a, _mm512_load_pd(row + 8 * v), _mm512_mul_pd(kj, d[v])));
	}

	/* h = [o ; x]^T W1 + b1; z = SiLU(h) */
	double *const h = A + CX_A_H, *const z = A + CX_A_Z;
	cx_gemv8(P + CX_O_W1, CX_DH, A + CX_A_MIN, CX_DM, h);
	for (int v = 0; v < CX_NVH; v++) {
		const __m512d hv = _mm512_add_pd(_mm512_load_pd(h + 8 * v), _mm512_load_pd(P + CX_O_B1 + 8 * v));
		const __m512d sg = cx_sigmoid(hv);
		_mm512_store_pd(h + 8 * v, hv);
		_mm512_store_pd(A + CX_A_SG + 8 * v, sg);
		_mm512_store_pd(z + 8 * v, _mm512_mul_pd(hv, sg));
	}

	/* u = z^T Wu; g = sigmoid(z . wg + bg); bc = z^T Wb */
	cx_gemv16(P + CX_O_WU, CX_DU, z, CX_DH, A + CX_A_U);
	__m512d gl = _mm512_load_pd(P + CX_O_BG);
	for (int v = 0; v < CX_NVH; v++) gl = _mm512_fmadd_pd(_mm512_load_pd(z + 8 * v), _mm512_load_pd(P + CX_O_WG + 8 * v), gl);
	_mm512_store_pd(A + CX_A_G, cx_sigmoid(cx_hsum(gl)));
	cx_gemv2(P + CX_O_WB, CX_DB, z, CX_DH, A + CX_A_BC);
	(void)one;
}

#endif
