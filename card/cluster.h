/*
 * cluster.h (F64 port): the 120-column cluster on the card (Stage 4), header only, 8 lanes of double. One thread per column (team.h map: two per
 * core), one shared embedding in vocabulary blocks (ETB), one consensus gain cg. Per token step every column sees the
 * same token; the lateral bus holds every column's bc from the previous step (stop-gradient). Reference: tools/cref.h.
 *
 * Step t, thread k (3 barriers):
 *   column forward (card/train.h saves)  ->  bus[t+1][d][k] = bc_k[d];  votes[t&1][k] = g_k u_k
 *   A: group leaders (k % 10 == 0) sum their 10 votes into gsum[t&1][k/10]
 *   B: every thread sums the 12 group vectors (same order, same bits): hs, xsh, hc = RMSNorm(hs) cg
 *      own blocks of the head: logits, local max m_k, exps, local sum s_k -> pairs[t&1][k]; l_tgt by every thread
 *   C: every thread reads the 120 pairs: M = max m_k, S = sum s_k 2^((m_k - M) log2 e); loss; own dl = (p - onehot) / L
 * Chunk backward: dhc partials over own blocks -> two-level sum (D, E) -> per step dhs, du_k = g_k dhs,
 *   dgl_k = (u_k . dhs) g_k (1 - g_k) -> the column backward (train.h) -> parameter gradients (local) -> own ETB blocks'
 *   gradient (head part local; input rows summed by the block owner after F) -> Adam (local; cg by thread 0) -> G.
 * Block ownership: thread k owns ETB blocks k and k + 120, and 240 + k when k < 16 (256 blocks).
 */
#ifndef CX_CLUSTER_H
#define CX_CLUSTER_H

#include "team.h"
#include "train.h"

#define CL_N CX_COLUMNS
#define CL_GROUPS 12
#define CL_GSIZE 10
#define CL_BUS (CX_DB * CX_NBUS)   /* floats per bus matrix */
#define CL_MAXBLK 3

typedef struct {
	/* shared */
	double *ETB, *GET, *m_et, *v_et;
	double *ES;         /* row-major shadow of ETB for the lookups; thread k refreshes its own blocks' rows after Adam */
	double *cg, *m_cg, *v_cg;
	double *bus;        /* (CX_LMAX + 1) x [16][128]: bus[t] is the input of step t */
	double *votes;      /* 2 x CL_N x 128 */
	double *gsum;       /* 2 x CL_GROUPS x 128 */
	double *pairs;     /* 2 x CL_N x 8: lane 0 m_k, lane 1 s_k */
	double *dhcp;       /* CL_N x L x 128 */
	double *gdhc;       /* CL_GROUPS x L x 128 */
	double *dep;        /* CL_N x L x 128 */
	double *loss;       /* L (thread 0) */
	volatile uint32_t *argmax;   /* 2 x 16: gen mode, the global argmax token of step t at argmax[(t & 1) * 16] */
	double *logits_out;           /* daemon mode: 4096 floats in a card huge page; each thread stores its blocks' raw logits */
	cx_barrier bar;
	int L, bus_on;
	/* per thread */
	cx_train tr[CL_N];
	double *hs[CL_N], *xsh[CL_N], *hcp[CL_N], *dhc[CL_N], *lg[CL_N], *tgtrow[CL_N];
	int nblk[CL_N], blk[CL_N][CL_MAXBLK];
	uint64_t epoch[CL_N];
} cluster;

static inline void cl_blocks(cluster *C, int k)
{
	C->nblk[k] = 0;
	C->blk[k][C->nblk[k]++] = k;
	C->blk[k][C->nblk[k]++] = k + CL_N;
	if (k < CX_VOCAB / 16 - 2 * CL_N) C->blk[k][C->nblk[k]++] = 2 * CL_N + k;
}

static inline int cl_owns(const cluster *C, int k, int g)
{
	for (int b = 0; b < C->nblk[k]; b++) if (C->blk[k][b] == g) return b;
	return -1;
}

static inline void cl_wait(cluster *C, int k) { cx_barrier_wait(&C->bar, k, ++C->epoch[k]); }

/* one cluster step for thread k: token id, next token (prefetch), target tgt; writes bus[t+1] */
static inline void cl_fwd_step(cluster *C, int k, int t, uint32_t id, uint32_t id_next, uint32_t tgt)
{
	cx_train *tr = &C->tr[k];
	const int L = C->L;
	const __m512d zero = _mm512_setzero_pd(), l2e = cx_set1(CX_LOG2E), invL = cx_set1(1.0f / L);
	const double *Bt = C->bus + (long)t * CL_BUS;
	double *bn = C->bus + (long)(t + 1) * CL_BUS;
	cx_pf_lines(C->ES + (long)tgt * CX_DE, CX_DE);
	cx_train_fwd_step(tr, t, id, id_next, Bt);
#ifdef CX_NOCOLL
	(void)tgt; (void)L; (void)zero; (void)l2e; (void)invL; (void)bn;
	return;
#endif
	const double *A = tr->col.act;
	if (C->bus_on) for (int d = 0; d < CX_DB; d++) bn[d * CX_NBUS + k] = A[CX_A_BC + d];
	double *vk = C->votes + ((long)(t & 1) * CL_N + k) * CX_DU;
	const __m512d g = _mm512_load_pd(A + CX_A_G);
	for (int v = 0; v < 16; v++) _mm512_store_pd(vk + 8 * v, _mm512_mul_pd(g, _mm512_load_pd(A + CX_A_U + 8 * v)));
	cl_wait(C, k);   /* A */
	if (k % CL_GSIZE == 0) {
		const double *v0 = C->votes + ((long)(t & 1) * CL_N + k) * CX_DU;
		cx_pf_lines(v0, CL_GSIZE * CX_DU);
		__m512d acc[16];
		for (int v = 0; v < 16; v++) acc[v] = _mm512_load_pd(v0 + 8 * v);
		for (int j = 1; j < CL_GSIZE; j++) for (int v = 0; v < 16; v++) acc[v] = _mm512_add_pd(acc[v], _mm512_load_pd(v0 + (long)j * CX_DU + 8 * v));
		double *gs = C->gsum + ((long)(t & 1) * CL_GROUPS + k / CL_GSIZE) * CX_DU;
		for (int v = 0; v < 16; v++) _mm512_store_pd(gs + 8 * v, acc[v]);
	}
	cl_wait(C, k);   /* B */
	const double *g0 = C->gsum + (long)(t & 1) * CL_GROUPS * CX_DU;
	cx_pf_lines(g0, CL_GROUPS * CX_DU);
	__m512d hs[16], ss = zero;
	for (int v = 0; v < 16; v++) hs[v] = _mm512_load_pd(g0 + 8 * v);
	for (int j = 1; j < CL_GROUPS; j++) for (int v = 0; v < 16; v++) hs[v] = _mm512_add_pd(hs[v], _mm512_load_pd(g0 + (long)j * CX_DU + 8 * v));
	for (int v = 0; v < 16; v++) { _mm512_store_pd(C->hs[k] + (long)t * CX_DU + 8 * v, hs[v]); ss = _mm512_fmadd_pd(hs[v], hs[v], ss); }
	const __m512d xsh = cx_rsqrt(_mm512_fmadd_pd(cx_hsum(ss), cx_set1(1.0 / CX_DU), cx_set1(CX_EPS)));
	_mm512_store_pd(C->xsh[k] + (long)t * 8, xsh);
	double *hc = C->hcp[k] + (long)t * CX_DU;
	for (int v = 0; v < 16; v++) _mm512_store_pd(hc + 8 * v, _mm512_mul_pd(_mm512_mul_pd(hs[v], xsh), _mm512_load_pd(C->cg + 8 * v)));
	/* own blocks of the head */
	double *lg = C->lg[k] + (long)t * CL_MAXBLK * 16;
	__m512d mk = cx_set1(-3.0e38f);
	for (int b = 0; b < C->nblk[k]; b++) {
		const double *e = C->ETB + (long)C->blk[k][b] * CX_BLOCK;
		__m512d acc0 = zero, acc1 = zero;
		for (int c = 0; c < CX_DE; c++) { const __m512d hb = CX_BC(hc + c); acc0 = _mm512_fmadd_pd(hb, _mm512_load_pd(e + c * 16), acc0); acc1 = _mm512_fmadd_pd(hb, _mm512_load_pd(e + c * 16 + 8), acc1); }
		_mm512_store_pd(lg + 16 * b, acc0);
		_mm512_store_pd(lg + 16 * b + 8, acc1);
		if (C->logits_out) { _mm512_store_pd(C->logits_out + (long)C->blk[k][b] * 16, acc0); _mm512_store_pd(C->logits_out + (long)C->blk[k][b] * 16 + 8, acc1); }
		mk = cx_max(mk, cx_max(acc0, acc1));
	}
	mk = cx_hmax(mk);
	__m512d sk = zero;
	for (int b = 0; b < C->nblk[k]; b++) {
		const __m512d e0 = cx_exp(_mm512_sub_pd(_mm512_load_pd(lg + 16 * b), mk)), e1 = cx_exp(_mm512_sub_pd(_mm512_load_pd(lg + 16 * b + 8), mk));
		_mm512_store_pd(lg + 16 * b, e0);
		_mm512_store_pd(lg + 16 * b + 8, e1);
		sk = _mm512_add_pd(sk, _mm512_add_pd(e0, e1));
	}
	sk = cx_hsum(sk);
	/* l_tgt = hc . ETB(tgt, :) */
	const double *trow = C->ES + (long)tgt * CX_DE;
	__m512d lt = zero;
	for (int v = 0; v < 16; v++) lt = _mm512_fmadd_pd(_mm512_load_pd(hc + 8 * v), _mm512_load_pd(trow + 8 * v), lt);
	lt = cx_hsum(lt);
	_mm512_store_pd(C->pairs + ((long)(t & 1) * CL_N + k) * 8, cx_blend(cx_blend(zero, 0x01, mk), 0x02, sk));
	cl_wait(C, k);   /* C */
	const double *p0 = C->pairs + (long)(t & 1) * CL_N * 8;
	cx_pf_lines(p0, CL_N * 8);
	double mb[CL_N + 8] __attribute__((aligned(64))), sb[CL_N + 8] __attribute__((aligned(64)));
	for (int j = 0; j < CL_N; j++) { mb[j] = p0[j * 8]; sb[j] = p0[j * 8 + 1]; }
	for (int j = CL_N; j < CL_N + 8; j++) { mb[j] = -1.0e300; sb[j] = 0; }
	__m512d M = _mm512_load_pd(mb);
	for (int v = 1; v < (CL_N + 8) / 8; v++) M = cx_max(M, _mm512_load_pd(mb + 8 * v));
	M = cx_hmax(M);
	__m512d S = zero;
	for (int v = 0; v < (CL_N + 8) / 8; v++) S = _mm512_fmadd_pd(_mm512_load_pd(sb + 8 * v), cx_exp(_mm512_sub_pd(_mm512_load_pd(mb + 8 * v), M)), S);
	S = cx_hsum(S);
	if (!k) cx_mask_store(C->loss + (t & ~7), (__mmask8)(1u << (t & 7)), _mm512_fmadd_pd(cx_log2(S), cx_set1(CX_LN2), _mm512_sub_pd(M, lt)));
	const __m512d scale = _mm512_mul_pd(_mm512_mul_pd(cx_exp(_mm512_sub_pd(mk, M)), cx_rcp(S)), invL);
	for (int b = 0; b < C->nblk[k]; b++) {
		__m512d d0 = _mm512_mul_pd(_mm512_load_pd(lg + 16 * b), scale), d1 = _mm512_mul_pd(_mm512_load_pd(lg + 16 * b + 8), scale);
		if (C->blk[k][b] == (int)(tgt >> 4)) { if (tgt & 8) d1 = _mm512_mask_sub_pd(d1, (__mmask8)(1u << (tgt & 7)), d1, invL); else d0 = _mm512_mask_sub_pd(d0, (__mmask8)(1u << (tgt & 7)), d0, invL); }
		_mm512_store_pd(lg + 16 * b, d0);
		_mm512_store_pd(lg + 16 * b + 8, d1);
	}
}

/* the chunk backward for thread k; ids[0..L] (ids[t] the input of step t) */
static inline void cl_bwd_chunk(cluster *C, int k, const uint32_t *ids, double b1, double b2, double lr_t, double eps_t)
{
	cx_train *tr = &C->tr[k];
	const int L = C->L;
	const __m512d zero = _mm512_setzero_pd(), one = cx_set1(1.0f);
	/* dhc partials over own blocks: dhcp[k][t][c] = sum_b sum_lane ETB[g_b][c] dl_t[b] */
	double *dp = C->dhcp + (long)k * L * CX_DU;
	for (int b = 0; b < C->nblk[k]; b++) cx_pf_lines(C->ETB + (long)C->blk[k][b] * CX_BLOCK, CX_BLOCK);
	for (int t = 0; t < L; t++) {
		const double *lg = C->lg[k] + (long)t * CL_MAXBLK * 16;
		for (int c = 0; c < CX_DE; c++) {
			__m512d acc = zero;
			for (int b = 0; b < C->nblk[k]; b++) { const double *eb = C->ETB + (long)C->blk[k][b] * CX_BLOCK + c * 16; acc = _mm512_fmadd_pd(_mm512_load_pd(eb), _mm512_load_pd(lg + 16 * b), acc); acc = _mm512_fmadd_pd(_mm512_load_pd(eb + 8), _mm512_load_pd(lg + 16 * b + 8), acc); }
			cx_mask_store(dp + (long)t * CX_DU + (c & ~7), (__mmask8)(1u << (c & 7)), cx_hsum(acc));
		}
	}
	cl_wait(C, k);   /* D */
	if (k % CL_GSIZE == 0) {
		const double *d0 = C->dhcp + (long)k * L * CX_DU;
		double *gd = C->gdhc + (long)(k / CL_GSIZE) * L * CX_DU;
		cx_pf_lines(d0, CL_GSIZE * L * CX_DU);
		for (int i = 0; i < L * CX_DU; i += 8) {
			__m512d acc = _mm512_load_pd(d0 + i);
			for (int j = 1; j < CL_GSIZE; j++) acc = _mm512_add_pd(acc, _mm512_load_pd(d0 + (long)j * L * CX_DU + i));
			_mm512_store_pd(gd + i, acc);
		}
	}
	cl_wait(C, k);   /* E */
	cx_pf_lines(C->gdhc, CL_GROUPS * L * CX_DU);
	for (int i = 0; i < L * CX_DU; i += 8) {
		__m512d acc = _mm512_load_pd(C->gdhc + i);
		for (int j = 1; j < CL_GROUPS; j++) acc = _mm512_add_pd(acc, _mm512_load_pd(C->gdhc + (long)j * L * CX_DU + i));
		_mm512_store_pd(C->dhc[k] + i, acc);
	}
	/* the steps in reverse */
	cx_train_chunk_begin(tr);
	cx_zero(tr->gcg, CX_DU);
	for (int t = L - 1; t >= 0; t--) {
		const double *A = tr->acts + (long)t * CX_NACT;
		cx_pf_lines(A, CX_NACT);
		cx_pf_lines(C->bus + (long)t * CL_BUS, CL_BUS);
		double dhs[CX_DU] __attribute__((aligned(64))), du[CX_DU] __attribute__((aligned(64)));
		cx_norm_bwd(C->dhc[k] + (long)t * CX_DU, C->hs[k] + (long)t * CX_DU, _mm512_load_pd(C->xsh[k] + (long)t * 8), C->cg, tr->gcg, dhs, 16, 1.0 / CX_DU);
		const __m512d g = _mm512_load_pd(A + CX_A_G);
		__m512d dg = zero;
		for (int v = 0; v < 16; v++) {
			const __m512d d = _mm512_load_pd(dhs + 8 * v);
			_mm512_store_pd(du + 8 * v, _mm512_mul_pd(g, d));
			dg = _mm512_fmadd_pd(_mm512_load_pd(A + CX_A_U + 8 * v), d, dg);
		}
		const __m512d dgl = _mm512_mul_pd(cx_hsum(dg), _mm512_mul_pd(g, _mm512_sub_pd(one, g)));
		cx_col_bwd_step(tr, t, C->bus + (long)t * CL_BUS, du, dgl);
	}
	cx_train_param_grads(tr);
	/* own blocks: GET[g][c] += sum_t hc_t[c] dl_t[g] (hc and dl private) */
	for (int b = 0; b < C->nblk[k]; b++) {
		double *G = C->GET + (long)C->blk[k][b] * CX_BLOCK;
		cx_pf_lines(G, CX_BLOCK);
		for (int tb = 0; tb < L; tb += 8) {
			__m512d d[8][2];
			for (int i = 0; i < 8; i++) { d[i][0] = _mm512_load_pd(C->lg[k] + (long)(tb + i) * CL_MAXBLK * 16 + 16 * b); d[i][1] = _mm512_load_pd(C->lg[k] + (long)(tb + i) * CL_MAXBLK * 16 + 16 * b + 8); }
			for (int c = 0; c < CX_DE; c++) {
				__m512d a0 = _mm512_load_pd(G + c * 16), a1 = _mm512_load_pd(G + c * 16 + 8);
				for (int i = 0; i < 8; i++) { const __m512d hb = CX_BC(C->hcp[k] + (long)(tb + i) * CX_DU + c); a0 = _mm512_fmadd_pd(hb, d[i][0], a0); a1 = _mm512_fmadd_pd(hb, d[i][1], a1); }
				_mm512_store_pd(G + c * 16, a0);
				_mm512_store_pd(G + c * 16 + 8, a1);
			}
		}
	}
	/* input-row gradients: publish de_t, then the block owners sum them */
	for (int t = 0; t < L; t++) cx_copy(C->dep + ((long)k * L + t) * CX_DU, tr->gsave + (long)t * CX_NG + CX_G_DE, CX_DU);
	cl_wait(C, k);   /* F */
	for (int t = 0; t < L; t++) {
		const int g = (int)(ids[t] >> 4);
		if (cl_owns(C, k, g) < 0) continue;
		__m512d acc[16];
		for (int v = 0; v < 16; v++) acc[v] = zero;
		for (int j = 0; j < CL_N; j++) {
			const double *d = C->dep + ((long)j * L + t) * CX_DU;
			if (j + 4 < CL_N) cx_pf_lines(C->dep + ((long)(j + 4) * L + t) * CX_DU, CX_DU);
			for (int v = 0; v < 16; v++) acc[v] = _mm512_add_pd(acc[v], _mm512_load_pd(d + 8 * v));
		}
		double de[CX_DU] __attribute__((aligned(64)));
		for (int v = 0; v < 16; v++) _mm512_store_pd(de + 8 * v, acc[v]);
		double *col = C->GET + (long)g * CX_BLOCK + (ids[t] & 15);
		for (int c = 0; c < CX_DE; c++) col[c * 16] += de[c];
	}
	/* Adam: own column, own blocks; cg by thread 0 (every thread holds the same gcg); lr_t < 0 = gradients only (gate) */
	if (lr_t < 0) { cl_wait(C, k); return; }
	cx_adam(tr->col.prm, tr->gprm, tr->gprm + ((CX_NPARAM + 7) & ~7), tr->gprm + 2 * ((CX_NPARAM + 7) & ~7), (CX_NPARAM + 7) & ~7, b1, b2, lr_t, eps_t);
	for (int b = 0; b < C->nblk[k]; b++) {
		const long o = (long)C->blk[k][b] * CX_BLOCK;
		cx_adam(C->ETB + o, C->GET + o, C->m_et + o, C->v_et + o, CX_BLOCK, b1, b2, lr_t, eps_t);
		cx_zero(C->GET + o, CX_BLOCK);
	}
	if (!k) cx_adam(C->cg, tr->gcg, C->m_cg, C->v_cg, CX_DU, b1, b2, lr_t, eps_t);
	/* refresh the shadow rows of the own blocks (the blocks are L2-hot after Adam) */
	for (int b = 0; b < C->nblk[k]; b++) {
		const int g = C->blk[k][b];
		for (int l = 0; l < 16; l++) {
			const double *src = C->ETB + (long)g * CX_BLOCK + l;
			double *row = C->ES + ((long)g * 16 + l) * CX_DE;
			for (int c = 0; c < CX_DE; c++) row[c] = src[c * 16];
		}
	}
	cl_wait(C, k);   /* G */
}

#endif

/* gen mode: called right after cl_fwd_step; publishes the argmax token of step t (ties: the lowest thread wins by a
 * second pass over the pairs) and returns it after one more barrier */
static inline uint32_t cl_argmax(cluster *C, int k, int t)
{
	const double *p0 = C->pairs + (long)(t & 1) * CL_N * 8;
	double M = -3.0e38f;
	int owner = -1;
	for (int j = 0; j < CL_N; j++) if (p0[j * 16] > M) { M = p0[j * 16]; owner = j; }
	if (owner == k) {
		/* which of my lanes holds M: the exps hold 2^((l - m_k) log2 e) = 1 at the max */
		const double *lg = C->lg[k] + (long)t * CL_MAXBLK * 16;
		uint32_t best = 0;
		double bv = -1;
		for (int b = 0; b < C->nblk[k]; b++)
			for (int l = 0; l < 16; l++) if (lg[16 * b + l] > bv) { bv = lg[16 * b + l]; best = (uint32_t)C->blk[k][b] * 16 + l; }
		C->argmax[(t & 1) * 16] = best;
	}
	cl_wait(C, k);
	return C->argmax[(t & 1) * 16];
}
