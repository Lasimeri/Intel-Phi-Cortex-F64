/*
 * train.h (F64 port): one column's training step on the VPU in double, 8 lanes. Same structure as the F32 version:
 * forward with per-step saves and the consensus norm hc = RMSNorm(u) cg; the tied head over a chunk of L steps;
 * softmax + cross-entropy + its gradient dl; the backward through the column and the fast-weight recurrence (dT carried
 * inside the chunk, T at the chunk start constant, the bus input stop-gradient); the parameter gradients as chunk-end
 * outer products over saved per-step vectors; the embedding gradient in two passes plus the input rows; Adam.
 *
 * The embedding is held in vocabulary blocks ETB[v / 16][c][v % 16] (256 blocks of 128 x 16 doubles = 16 KiB); a
 * block row is two vectors. Saves use plain stores into lines prefetched for writing one step ahead; data consumed
 * once is evicted from L2 (clevict1) after use (measured on the F32 port).
 */
#ifndef CX_TRAIN_H
#define CX_TRAIN_H

#include "col.h"

#define CX_LMAX 32
#define CX_LN2 0.6931471805599453
#ifndef CX_EVICT
#define CX_EVICT 1
#endif
#define CX_ETB(v, c) ((((long)(v) >> 4) * CX_DE + (c)) * 16 + ((v) & 15))
#define CX_BLOCK (CX_DE * 16)   /* doubles per vocabulary block */

typedef struct {
	cx_col col;      /* col.act = the hot block */
	double *acts;    /* CX_LMAX x CX_NACT */
	double *Ts;      /* CX_LMAX x CX_NT: T before each step */
	double *ET;      /* ETB */
	const double *ES;/* optional row-major shadow E[v][c] of ETB: 16-line input rows */
	double *cg;      /* consensus gain, 128 */
	double *hc;      /* CX_LMAX x 128 */
	double *logits;  /* CX_LMAX x CX_VOCAB; after cx_head_loss: dl = (softmax - onehot) / L */
	double *loss;    /* CX_LMAX */
	double *ebuf;    /* 128 */
	double *dhc;     /* CX_LMAX x 128 */
	double *dT;      /* 64 x 64, carried */
	double *gsave;   /* CX_LMAX x CX_NG */
	double *gprm;    /* CX_NPARAM */
	double *gcg;     /* 128 */
	double *gb1;     /* CX_DH */
	double *ggain;   /* 80 */
	double *gwg;     /* CX_DH */
	double *gbg;     /* 8, lane 0 */
	double *GET;     /* ETB-layout embedding gradient */
	double *tmp;     /* 1024 doubles of scratch */
	int L;
} cx_train;

static inline void cx_evict1(const void *p) { if (CX_EVICT) __asm__ volatile("clevict1 %0" : : "m"(*(const char *)p)); }
static inline void cx_pfe0(const void *p) { __asm__ volatile("vprefetche0 %0" : : "m"(*(const char *)p)); }

static inline void cx_copy(double *dst, const double *src, int n)
{
	for (int i = 0; i < n; i += 8) _mm512_store_pd(dst + i, _mm512_load_pd(src + i));
}
static inline void cx_copy_evict(double *dst, const double *src, int n)
{
	cx_copy(dst, src, n);
	for (int i = 0; i < n; i += 8) cx_evict1(dst + i);
}
static inline void cx_pf_lines(const double *p, int n) { for (int i = 0; i < n; i += 8) cx_pf0(p + i); }
static inline void cx_pfe_lines(const double *p, int n) { for (int i = 0; i < n; i += 8) cx_pfe0(p + i); }
static inline void cx_zero(double *p, long n) { const __m512d z = _mm512_setzero_pd(); for (long i = 0; i < n; i += 8) _mm512_store_pd(p + i, z); }

/* ---- forward ---- */

static inline void cx_gather_prefetch(const double *ETB, uint32_t id) { cx_pf_lines(ETB + ((long)id >> 4) * CX_BLOCK, CX_BLOCK); }

static inline void cx_gather_row(const double *ETB, uint32_t id, double *e)
{
	const double *b = ETB + ((long)id >> 4) * CX_BLOCK + (id & 15);
	for (int c = 0; c < CX_DE; c++) e[c] = b[c * 16];
	for (int c = 0; c < CX_DE; c += 4) cx_evict1(b + c * 16);   /* 16 doubles = 2 lines per c */
}

static inline void cx_train_fwd_step(cx_train *tr, int t, uint32_t id, uint32_t id_next, const double *Bt)
{
	double *const A = tr->col.act;
#ifndef CX_NOSNAP
	cx_copy_evict(tr->Ts + (long)t * CX_NT, tr->col.T, CX_NT);
#endif
	if (tr->ES) {
		cx_copy(tr->ebuf, tr->ES + (long)id * CX_DE, CX_DE);
		cx_pf_lines(tr->ES + (long)id_next * CX_DE, CX_DE);
	} else {
		cx_gather_row(tr->ET, id, tr->ebuf);
		cx_gather_prefetch(tr->ET, id_next);
	}
	if (t + 1 < CX_LMAX) { cx_pfe_lines(tr->Ts + (long)(t + 1) * CX_NT, CX_NT); cx_pfe_lines(tr->acts + (long)(t + 1) * CX_NACT, CX_NACT); }
	cx_col_step(&tr->col, tr->ebuf, Bt);
	cx_copy(A + CX_A_E, tr->ebuf, CX_DE);
	__m512d u[16], ss = _mm512_setzero_pd();
	for (int v = 0; v < 16; v++) { u[v] = _mm512_load_pd(A + CX_A_U + 8 * v); ss = _mm512_fmadd_pd(u[v], u[v], ss); }
	const __m512d xsu = cx_rsqrt(_mm512_fmadd_pd(cx_hsum(ss), cx_set1(1.0 / CX_DU), cx_set1(CX_EPS)));
	_mm512_store_pd(A + CX_A_XSU, xsu);
	for (int v = 0; v < 16; v++) {
		const __m512d h = _mm512_mul_pd(_mm512_mul_pd(u[v], xsu), _mm512_load_pd(tr->cg + 8 * v));
		_mm512_store_pd(A + CX_A_HC + 8 * v, h);
		_mm512_store_pd(tr->hc + (long)t * CX_DU + 8 * v, h);
	}
	cx_copy_evict(tr->acts + (long)t * CX_NACT, A, CX_NACT);
}

/* logits[t][v] = sum_c hc[t][c] ETB(v, c): tiles of 8 steps x 1 block (2 vectors each, 16 accumulators), ETB streamed L/8 times */
static inline void cx_head_logits(cx_train *tr)
{
	for (int tb = 0; tb < tr->L; tb += 8)
		for (int g = 0; g < CX_VOCAB / 16; g++) {
			__m512d acc[8][2];
			_Pragma("unroll") for (int i = 0; i < 8; i++) { acc[i][0] = _mm512_setzero_pd(); acc[i][1] = _mm512_setzero_pd(); }
			const double *hc0 = tr->hc + (long)tb * CX_DU, *b0 = tr->ET + (long)g * CX_BLOCK;
			for (int c = 0; c < CX_DE; c++) {
				if (c + 8 < CX_DE) { cx_pf0(b0 + (c + 8) * 16); cx_pf0(b0 + (c + 8) * 16 + 8); }
				const __m512d e0 = _mm512_load_pd(b0 + c * 16), e1 = _mm512_load_pd(b0 + c * 16 + 8);
				_Pragma("unroll") for (int i = 0; i < 8; i++) {
					const __m512d hb = CX_BC(hc0 + i * CX_DU + c);
					acc[i][0] = _mm512_fmadd_pd(hb, e0, acc[i][0]);
					acc[i][1] = _mm512_fmadd_pd(hb, e1, acc[i][1]);
				}
				cx_evict1(b0 + c * 16);
				cx_evict1(b0 + c * 16 + 8);
			}
			_Pragma("unroll") for (int i = 0; i < 8; i++) {
				double *o = tr->logits + (long)(tb + i) * CX_VOCAB + g * 16;
				_mm512_store_pd(o, acc[i][0]);
				_mm512_store_pd(o + 8, acc[i][1]);
				cx_evict1(o);
				cx_evict1(o + 8);
			}
		}
}

/* per step: loss[t] = logsumexp(logits) - logits[tgt]; logits[t] := (softmax - onehot(tgt)) / L */
static inline void cx_head_loss(cx_train *tr, const uint32_t *tgt)
{
	const __m512d l2e = cx_set1(1.4426950408889634), ln2 = cx_set1(CX_LN2), invL = cx_set1(1.0 / tr->L);
	for (int t = 0; t < tr->L; t++) {
		double *const l = tr->logits + (long)t * CX_VOCAB;
		for (int v = 0; v < 8; v++) cx_pf0(l + 8 * v);
		__m512d m = cx_set1(-1.0e300);
		for (int v = 0; v < CX_VOCAB / 8; v++) {
			if (v + 8 < CX_VOCAB / 8) cx_pf0(l + 8 * (v + 8));
			m = cx_max(m, _mm512_load_pd(l + 8 * v));
		}
		m = cx_hmax(m);
		const __m512d ltgt = CX_BC(l + tgt[t]);
		__m512d sum = _mm512_setzero_pd();
		for (int v = 0; v < CX_VOCAB / 8; v++) {
			const __m512d e = cx_exp(_mm512_sub_pd(_mm512_load_pd(l + 8 * v), m));
			_mm512_store_pd(l + 8 * v, e);
			sum = _mm512_add_pd(sum, e);
		}
		sum = cx_hsum(sum);
		const __m512d lossv = _mm512_fmadd_pd(cx_log2(sum), ln2, _mm512_sub_pd(m, ltgt));
		cx_mask_store(tr->loss + (t & ~7), (__mmask8)(1u << (t & 7)), lossv);
		const __m512d scale = _mm512_mul_pd(cx_rcp(sum), invL);
		for (int v = 0; v < CX_VOCAB / 8; v++) _mm512_store_pd(l + 8 * v, _mm512_mul_pd(_mm512_load_pd(l + 8 * v), scale));
		double *const lt = l + (tgt[t] & ~7u);
		const __m512d d = _mm512_load_pd(lt);
		_mm512_store_pd(lt, _mm512_mask_sub_pd(d, (__mmask8)(1u << (tgt[t] & 7)), d, invL));
		for (int v = 0; v < CX_VOCAB / 8; v++) cx_evict1(l + 8 * v);
	}
}

/* ---- backward ---- */

/* dhc[t][c] = sum_v ETB(v, c) dl[t][v]: tiles of 4 steps x 4 c (16 accumulators, each over the block row's 2 vectors) */
static inline void cx_head_dhc(cx_train *tr)
{
	for (int tb = 0; tb < tr->L; tb += 4)
		for (int c = 0; c < CX_DE; c += 4) {
			__m512d acc[4][4];
			_Pragma("unroll") for (int i = 0; i < 4; i++) _Pragma("unroll") for (int j = 0; j < 4; j++) acc[i][j] = _mm512_setzero_pd();
			const double *dl0 = tr->logits + (long)tb * CX_VOCAB, *e0 = tr->ET + c * 16;
			for (int g = 0; g < CX_VOCAB / 16; g++, e0 += CX_BLOCK) {
				if (g + 8 < CX_VOCAB / 16) _Pragma("unroll") for (int j = 0; j < 8; j++) cx_pf0(e0 + 8 * CX_BLOCK + 8 * j);
				_Pragma("unroll") for (int i = 0; i < 4; i++) {
					const __m512d d0 = _mm512_load_pd(dl0 + (long)i * CX_VOCAB + g * 16), d1 = _mm512_load_pd(dl0 + (long)i * CX_VOCAB + g * 16 + 8);
					_Pragma("unroll") for (int j = 0; j < 4; j++) {
						acc[i][j] = _mm512_fmadd_pd(_mm512_load_pd(e0 + 16 * j), d0, acc[i][j]);
						acc[i][j] = _mm512_fmadd_pd(_mm512_load_pd(e0 + 16 * j + 8), d1, acc[i][j]);
					}
				}
				_Pragma("unroll") for (int j = 0; j < 8; j++) cx_evict1(e0 + 8 * j);
			}
			_Pragma("unroll") for (int i = 0; i < 4; i++) {
				double *o = tr->dhc + (long)(tb + i) * CX_DU + (c & ~7);
				_Pragma("unroll") for (int j = 0; j < 4; j++) cx_mask_store(o, (__mmask8)(1u << ((c + j) & 7)), cx_hsum(acc[i][j]));
			}
		}
}

/* GET(v, c) += sum_t hc[t][c] dl[t][v]: per block and 8-step tile, dl rows in registers, the block RMW once */
static inline void cx_head_dE(cx_train *tr)
{
	for (int g = 0; g < CX_VOCAB / 16; g++) {
		double *G = tr->GET + (long)g * CX_BLOCK;
		cx_pf_lines(G, CX_BLOCK);
		for (int tb = 0; tb < tr->L; tb += 8) {
			__m512d d[8][2];
			_Pragma("unroll") for (int i = 0; i < 8; i++) { d[i][0] = _mm512_load_pd(tr->logits + (long)(tb + i) * CX_VOCAB + g * 16); d[i][1] = _mm512_load_pd(tr->logits + (long)(tb + i) * CX_VOCAB + g * 16 + 8); }
			const double *hc0 = tr->hc + (long)tb * CX_DU;
			for (int c = 0; c < CX_DE; c++) {
				__m512d a0 = _mm512_load_pd(G + c * 16), a1 = _mm512_load_pd(G + c * 16 + 8);
				_Pragma("unroll") for (int i = 0; i < 8; i++) { const __m512d hb = CX_BC(hc0 + i * CX_DU + c); a0 = _mm512_fmadd_pd(hb, d[i][0], a0); a1 = _mm512_fmadd_pd(hb, d[i][1], a1); }
				_mm512_store_pd(G + c * 16, a0);
				_mm512_store_pd(G + c * 16 + 8, a1);
			}
		}
		for (int c = 0; c < CX_DE; c++) { cx_evict1(G + c * 16); cx_evict1(G + c * 16 + 8); }
		for (int i = 0; i < tr->L; i++) { cx_evict1(tr->logits + (long)i * CX_VOCAB + g * 16); cx_evict1(tr->logits + (long)i * CX_VOCAB + g * 16 + 8); }
	}
}

/* y[r] = W[r] . x over 8 NV lanes, rows ld apart, r < R (R a multiple of 8): dot form, one lane reduction per row */
#define CX_GEMVT(NV)                                                                                                 \
	static inline void cx_gemvt##NV(const double *W, long ld, const double *x, int R, double *y)                     \
	{                                                                                                                \
		__m512d xv[NV];                                                                                              \
		_Pragma("unroll") for (int v = 0; v < NV; v++) xv[v] = _mm512_load_pd(x + 8 * v);                            \
		for (int r = 0; r < R; r++) {                                                                                \
			const double *w = W + (long)r * ld;                                                                      \
			_Pragma("unroll") for (int v = 0; v < NV; v++) cx_pf0(w + 2 * ld + 8 * v);                               \
			__m512d acc = _mm512_mul_pd(xv[0], _mm512_load_pd(w));                                                   \
			_Pragma("unroll") for (int v = 1; v < NV; v++) acc = _mm512_fmadd_pd(xv[v], _mm512_load_pd(w + 8 * v), acc); \
			cx_mask_store(y + (r & ~7), (__mmask8)(1u << (r & 7)), cx_hsum(acc));                                    \
		}                                                                                                            \
	}
CX_GEMVT(2)
CX_GEMVT(8)
CX_GEMVT(13)
CX_GEMVT(16)

/* RMSNorm backward with gain: given dx (on x = n gain, n = xin xs), writes dxin and adds dx n into ggain */
static inline void cx_norm_bwd(const double *dx, const double *xin, __m512d xs, const double *gain, double *ggain, double *dxin, int NV, double invN)
{
	__m512d dn[16], n[16], dot = _mm512_setzero_pd();
	for (int v = 0; v < NV; v++) {
		n[v] = _mm512_mul_pd(_mm512_load_pd(xin + 8 * v), xs);
		const __m512d d = _mm512_load_pd(dx + 8 * v);
		dn[v] = _mm512_mul_pd(d, _mm512_load_pd(gain + 8 * v));
		_mm512_store_pd(ggain + 8 * v, _mm512_fmadd_pd(d, n[v], _mm512_load_pd(ggain + 8 * v)));
		dot = _mm512_fmadd_pd(dn[v], n[v], dot);
	}
	dot = _mm512_mul_pd(cx_hsum(dot), cx_set1(invN));
	for (int v = 0; v < NV; v++) _mm512_store_pd(dxin + 8 * v, _mm512_mul_pd(xs, cx_fnmadd(n[v], dot, dn[v])));
}

/* column backward of step t given du (128) and dgl (all lanes) */
static inline void cx_col_bwd_step(cx_train *tr, int t, const double *Bt, const double *du, __m512d dgl)
{
	const double *P = tr->col.prm, *A = tr->acts + (long)t * CX_NACT, *T0 = tr->Ts + (long)t * CX_NT;
	double *const G = tr->gsave + (long)t * CX_NG, *const S = tr->tmp, *const dT = tr->dT;
	cx_pf_lines(A, CX_NACT);
	cx_pf_lines(T0, CX_NT);
	if (t) cx_pf_lines(tr->acts + (long)(t - 1) * CX_NACT, CX_NACT);
	const __m512d zero = _mm512_setzero_pd(), one = cx_set1(1.0);

	cx_copy(G + CX_G_DU, du, CX_DU);
	/* dz = Wu du (+ wg dgl); dh = dz (sg + h sg (1 - sg)); gb1 += dh; gwg += z dgl; gbg += dgl */
	double *const dz = S, *const dh = G + CX_G_DH;
	cx_gemvt16(P + CX_O_WU, CX_DU, G + CX_G_DU, CX_DH, dz);
	for (int v = 0; v < CX_NVH; v++) {
		_mm512_store_pd(dz + 8 * v, _mm512_fmadd_pd(dgl, _mm512_load_pd(P + CX_O_WG + 8 * v), _mm512_load_pd(dz + 8 * v)));
		_mm512_store_pd(tr->gwg + 8 * v, _mm512_fmadd_pd(dgl, _mm512_load_pd(A + CX_A_Z + 8 * v), _mm512_load_pd(tr->gwg + 8 * v)));
	}
	_mm512_store_pd(tr->gbg, _mm512_add_pd(_mm512_load_pd(tr->gbg), dgl));
	for (int v = 0; v < CX_NVH; v++) {
		const __m512d sg = _mm512_load_pd(A + CX_A_SG + 8 * v), h = _mm512_load_pd(A + CX_A_H + 8 * v);
		const __m512d f = _mm512_fmadd_pd(_mm512_mul_pd(h, sg), _mm512_sub_pd(one, sg), sg);
		const __m512d d = _mm512_mul_pd(_mm512_load_pd(dz + 8 * v), f);
		_mm512_store_pd(dh + 8 * v, d);
		_mm512_store_pd(tr->gb1 + 8 * v, _mm512_add_pd(_mm512_load_pd(tr->gb1 + 8 * v), d));
	}
	/* dmin = W1 dh: do = dmin[0..64), dx = dmin[64..144) */
	double *const dmin = S + 64;   /* 144 */
	cx_gemvt8(P + CX_O_W1, CX_DH, dh, CX_DM, dmin);
	const double *dO = dmin;
	/* fast weights */
	const __m512d a = _mm512_load_pd(A + CX_A_AV), b = _mm512_load_pd(A + CX_A_BV), kq = _mm512_load_pd(A + CX_A_KQ);
	const double *k = A + CX_A_K, *y = A + CX_A_Y, *vv = y + 128, *dd0 = A + CX_A_D, *Sq = A + CX_A_SQ, *Sk = A + CX_A_SK;
	__m512d da = zero, dkq = zero, dSq[8], dd[8];
	for (int v = 0; v < 8; v++) {
		const __m512d o = _mm512_load_pd(dO + 8 * v);
		da = _mm512_fmadd_pd(o, _mm512_load_pd(Sq + 8 * v), da);
		dkq = _mm512_fmadd_pd(o, _mm512_load_pd(dd0 + 8 * v), dkq);
		dSq[v] = _mm512_mul_pd(a, o);
		dd[v] = _mm512_mul_pd(o, kq);
	}
	for (int j = 0; j < CX_DS; j++) {   /* sweep 1: da += <dT, T0>, dd += k_j dT[j] */
		const double *g = dT + j * CX_DS, *t0 = T0 + j * CX_DS;
		const __m512d kj = CX_BC(k + j);
		for (int v = 0; v < 8; v++) {
			const __m512d gv = _mm512_load_pd(g + 8 * v);
			da = _mm512_fmadd_pd(gv, _mm512_load_pd(t0 + 8 * v), da);
			dd[v] = _mm512_fmadd_pd(kj, gv, dd[v]);
		}
	}
	dkq = cx_hsum(dkq);
	const __m512d ab = _mm512_mul_pd(a, b);
	__m512d db = zero, dSk[8], dv[8];
	for (int v = 0; v < 8; v++) {
		const __m512d skv = _mm512_load_pd(Sk + 8 * v);
		dv[v] = _mm512_mul_pd(b, dd[v]);
		db = _mm512_fmadd_pd(dd[v], cx_fnmadd(a, skv, _mm512_load_pd(vv + 8 * v)), db);
		da = cx_fnmadd(dd[v], _mm512_mul_pd(b, skv), da);
		dSk[v] = _mm512_mul_pd(_mm512_sub_pd(zero, ab), dd[v]);
	}
	da = cx_hsum(da);
	db = cx_hsum(db);
	double *const dk = S + 208, *const dq = S + 272;   /* 64 each */
	double dvec[CX_DS] __attribute__((aligned(64)));
	for (int v = 0; v < 8; v++) _mm512_store_pd(dvec + 8 * v, _mm512_load_pd(dd0 + 8 * v));
	for (int j = 0; j < CX_DS; j++) {   /* sweep 2: dk_j, dq_j, and dT[j] := a dT[j] + q_j dSq + k_j dSk */
		double *g = dT + j * CX_DS;
		const double *t0 = T0 + j * CX_DS;
		const __m512d kj = CX_BC(k + j), qj = CX_BC(y + j);
		__m512d ak = zero, aq = zero;
		for (int v = 0; v < 8; v++) {
			const __m512d gv = _mm512_load_pd(g + 8 * v), tv = _mm512_load_pd(t0 + 8 * v);
			ak = _mm512_fmadd_pd(gv, _mm512_load_pd(dvec + 8 * v), ak);
			ak = _mm512_fmadd_pd(tv, dSk[v], ak);
			aq = _mm512_fmadd_pd(tv, dSq[v], aq);
			_mm512_store_pd(g + 8 * v, _mm512_fmadd_pd(kj, dSk[v], _mm512_fmadd_pd(qj, dSq[v], _mm512_mul_pd(a, gv))));
		}
		cx_mask_store(dk + (j & ~7), (__mmask8)(1u << (j & 7)), _mm512_fmadd_pd(dkq, qj, cx_hsum(ak)));
		cx_mask_store(dq + (j & ~7), (__mmask8)(1u << (j & 7)), _mm512_fmadd_pd(dkq, kj, cx_hsum(aq)));
	}
	__m512d kdk = zero;
	for (int v = 0; v < 8; v++) kdk = _mm512_fmadd_pd(_mm512_load_pd(dk + 8 * v), _mm512_load_pd(k + 8 * v), kdk);
	kdk = cx_hsum(kdk);
	const __m512d ks = _mm512_load_pd(A + CX_A_KS);
	double *const dy = G + CX_G_DY;
	for (int v = 0; v < 8; v++) {
		_mm512_store_pd(dy + 8 * v, _mm512_load_pd(dq + 8 * v));
		_mm512_store_pd(dy + 64 + 8 * v, _mm512_mul_pd(ks, cx_fnmadd(_mm512_load_pd(k + 8 * v), kdk, _mm512_load_pd(dk + 8 * v))));
		_mm512_store_pd(dy + 128 + 8 * v, dv[v]);
	}
	const __m512d dap = _mm512_mul_pd(da, _mm512_mul_pd(a, _mm512_sub_pd(one, a))), dbp = _mm512_mul_pd(db, _mm512_mul_pd(b, _mm512_sub_pd(one, b)));
	_mm512_store_pd(dy + 192, cx_blend(cx_blend(zero, 0x01, dap), 0x02, dbp));
	_mm512_store_pd(dy + 200, zero);
	/* dx = dmin[64..) + W dy (two 13-vector halves of the 208-wide rows) */
	double *const dxw = S + 336, *const dxw2 = S + 416;   /* 80 each */
	cx_gemvt13(P + CX_O_W, CX_NW, dy, CX_DX, dxw);
	cx_gemvt13(P + CX_O_W + 104, CX_NW, dy + 104, CX_DX, dxw2);
	for (int v = 0; v < 10; v++) _mm512_store_pd(dxw + 8 * v, _mm512_add_pd(_mm512_add_pd(_mm512_load_pd(dxw + 8 * v), _mm512_load_pd(dxw2 + 8 * v)), _mm512_load_pd(dmin + 64 + 8 * v)));
	/* x = RMSNorm([p ; r]) gain */
	double *const xin = S + 496;   /* 80 */
	cx_copy(xin, A + CX_A_P, CX_DP);
	_mm512_store_pd(xin + CX_DP, _mm512_load_pd(A + CX_A_R));
	_mm512_store_pd(xin + CX_DP + 8, _mm512_load_pd(A + CX_A_R + 8));
	double *const dxin = S + 576;   /* 80: dp (64) then dr (16) */
	cx_norm_bwd(dxw, xin, _mm512_load_pd(A + CX_A_XS), P + CX_O_GAIN, tr->ggain, dxin, 10, 1.0 / CX_DX);
	/* bus attention: dalpha = Bt^T dr; ds = alpha (dalpha - alpha . dalpha); dqb = Bt ds */
	double *const dalpha = S + 656;   /* 128 */
	cx_gemv16(Bt, CX_NBUS, dxin + 64, CX_DB, dalpha);
	__m512d adot = zero;
	for (int v = 0; v < 16; v++) adot = _mm512_fmadd_pd(_mm512_load_pd(A + CX_A_ALPHA + 8 * v), _mm512_load_pd(dalpha + 8 * v), adot);
	adot = cx_hsum(adot);
	for (int v = 0; v < 16; v++) _mm512_store_pd(dalpha + 8 * v, _mm512_mul_pd(_mm512_load_pd(A + CX_A_ALPHA + 8 * v), _mm512_sub_pd(_mm512_load_pd(dalpha + 8 * v), adot)));
	cx_gemvt16(Bt, CX_NBUS, dalpha, CX_DB, G + CX_G_DQB);
	/* dp = dxin[0..64) + Wqb dqb; de = P_in dp */
	double *const dp = G + CX_G_DP;
	cx_gemvt2(P + CX_O_WQB, CX_DB, G + CX_G_DQB, CX_DP, dp);
	for (int v = 0; v < 8; v++) _mm512_store_pd(dp + 8 * v, _mm512_add_pd(_mm512_load_pd(dp + 8 * v), _mm512_load_pd(dxin + 8 * v)));
	cx_gemvt8(P + CX_O_PIN, CX_DP, dp, CX_DE, G + CX_G_DE);
	for (int i = 0; i < CX_NT; i += 8) cx_evict1(T0 + i);
}

/* Gm[k][..] = sum_t a[t][k] g[t][8 NV], output rows gld apart */
#define CX_OUTER(NV)                                                                                                 \
	static inline void cx_outer##NV(double *Gm, long gld, const double *a, long as, const double *g, long gs, int K, int L) \
	{                                                                                                                \
		for (int k = 0; k < K; k++) {                                                                                \
			__m512d acc[NV];                                                                                         \
			_Pragma("unroll") for (int v = 0; v < NV; v++) acc[v] = _mm512_setzero_pd();                             \
			for (int t = 0; t < L; t++) {                                                                            \
				const __m512d ab = CX_BC(a + t * as + k);                                                            \
				_Pragma("unroll") for (int v = 0; v < NV; v++) acc[v] = _mm512_fmadd_pd(ab, _mm512_load_pd(g + t * gs + 8 * v), acc[v]); \
			}                                                                                                        \
			_Pragma("unroll") for (int v = 0; v < NV; v++) _mm512_store_pd(Gm + (long)k * gld + 8 * v, acc[v]);      \
		}                                                                                                            \
	}
CX_OUTER(2)
CX_OUTER(8)
CX_OUTER(13)
CX_OUTER(16)

static inline void cx_train_param_grads(cx_train *tr)
{
	const int L = tr->L;
	double *gp = tr->gprm;
	cx_outer8(gp + CX_O_PIN, CX_DP, tr->acts + CX_A_E, CX_NACT, tr->gsave + CX_G_DP, CX_NG, CX_DE, L);
	cx_outer2(gp + CX_O_WQB, CX_DB, tr->acts + CX_A_P, CX_NACT, tr->gsave + CX_G_DQB, CX_NG, CX_DP, L);
	cx_copy(gp + CX_O_GAIN, tr->ggain, CX_DX);
	cx_outer13(gp + CX_O_W, CX_NW, tr->acts + CX_A_MIN + CX_DS, CX_NACT, tr->gsave + CX_G_DY, CX_NG, CX_DX, L);
	cx_outer13(gp + CX_O_W + 104, CX_NW, tr->acts + CX_A_MIN + CX_DS, CX_NACT, tr->gsave + CX_G_DY + 104, CX_NG, CX_DX, L);
	cx_outer8(gp + CX_O_W1, CX_DH, tr->acts + CX_A_MIN, CX_NACT, tr->gsave + CX_G_DH, CX_NG, CX_DM, L);
	cx_copy(gp + CX_O_B1, tr->gb1, CX_DH);
	cx_outer16(gp + CX_O_WU, CX_DU, tr->acts + CX_A_Z, CX_NACT, tr->gsave + CX_G_DU, CX_NG, CX_DH, L);
	cx_copy(gp + CX_O_WG, tr->gwg, CX_DH);
	cx_copy(gp + CX_O_BG, tr->gbg, 8);
	cx_zero(gp + CX_O_BG + 8, 8);
	cx_zero(gp + CX_O_WB, CX_NPARAM - CX_O_WB);   /* Wb: the bus is stop-gradient */
}

static inline void cx_train_input_grads(cx_train *tr, const uint32_t *ids)
{
	for (int t = 0; t < tr->L; t++) {
		double *b = tr->GET + ((long)ids[t] >> 4) * CX_BLOCK + (ids[t] & 15);
		const double *de = tr->gsave + (long)t * CX_NG + CX_G_DE;
		for (int c = 0; c < CX_DE; c += 4) cx_pf0(b + c * 16);
		for (int c = 0; c < CX_DE; c++) b[c * 16] += de[c];
	}
}

static inline void cx_train_chunk_begin(cx_train *tr)
{
	cx_zero(tr->dT, CX_NT);
	cx_zero(tr->gcg, CX_DU);
	cx_zero(tr->gb1, CX_DH);
	cx_zero(tr->ggain, CX_DX);
	cx_zero(tr->gwg, CX_DH);
	cx_zero(tr->gbg, 8);
}

/* ---- optimizer and the single-column bus ---- */

static inline void cx_adam(double *p, const double *g, double *m, double *v, long n, double b1, double b2, double lr_t, double eps_t)
{
	const __m512d B1 = cx_set1(b1), B2 = cx_set1(b2), O1 = cx_set1(1.0 - b1), O2 = cx_set1(1.0 - b2);
	const __m512d LR = cx_set1(lr_t), EPS = cx_set1(eps_t), TINY = cx_set1(1e-30);   /* keeps s inside the F32 normal range that the cx_rsqrt / cx_rcp seeds need (1e-300 gave NaN for zero-gradient parameters) */
	for (long i = 0; i < n; i += 8) {
		if (i + 32 < n) { cx_pf0(p + i + 32); cx_pf0(g + i + 32); cx_pf0(m + i + 32); cx_pf0(v + i + 32); }
		const __m512d gv = _mm512_load_pd(g + i);
		const __m512d mv = _mm512_fmadd_pd(O1, gv, _mm512_mul_pd(B1, _mm512_load_pd(m + i)));
		const __m512d vv = _mm512_fmadd_pd(O2, _mm512_mul_pd(gv, gv), _mm512_mul_pd(B2, _mm512_load_pd(v + i)));
		_mm512_store_pd(m + i, mv);
		_mm512_store_pd(v + i, vv);
		const __m512d s = _mm512_add_pd(vv, TINY);
		const __m512d den = _mm512_fmadd_pd(s, cx_rsqrt(s), EPS);
		_mm512_store_pd(p + i, cx_fnmadd(LR, _mm512_mul_pd(mv, cx_rcp(den)), _mm512_load_pd(p + i)));
	}
}

static inline void cx_bus_set0(double *Bt, const double *bc)
{
	for (int d = 0; d < CX_DB; d++) Bt[d * CX_NBUS] = bc[d];
}

/* single-column step backward: the consensus norm then the column with dgl = 0 */
static inline void cx_train_bwd_step(cx_train *tr, int t, const double *Bt)
{
	const double *A = tr->acts + (long)t * CX_NACT;
	cx_pf_lines(A, CX_NACT);
	double du[CX_DU] __attribute__((aligned(64)));
	cx_norm_bwd(tr->dhc + (long)t * CX_DU, A + CX_A_U, _mm512_load_pd(A + CX_A_XSU), tr->cg, tr->gcg, du, 16, 1.0 / CX_DU);
	cx_col_bwd_step(tr, t, Bt, du, _mm512_setzero_pd());
}

#endif
