/*
 * cref.h: host double reference of the CLUSTER training step (Stage 4): N columns (tools/ref.h per column), one shared
 * embedding E and consensus gain cg, the same token to every column per step.
 *
 *   per step t:  column c forward on token id_t with the bus Bt (slot j = column j's bc from step t-1, zero at start)
 *                hs = sum_c g_c u_c      hc = RMSNorm(hs) cg      logits = E hc      loss = CE(next token)
 *                bus for t+1: slot c = bc_c (one step delayed, stop-gradient)
 *   backward:    dhc = E^T dl;  dhs = RMSNorm backward (dcg += dhc n);  du_c = g_c dhs;  dgl_c = (u_c . dhs) g_c (1 - g_c)
 *                then each column's own backward (ref_col_backward), dT_c carried per column inside the chunk.
 *
 * Bus modes: CREF_BUS_LIVE (the columns' own bc, as cortexd runs), CREF_BUS_ZERO (all zero, the card's CX_BUS=0) or CREF_BUS_EXT (an external matrix per step, for the
 * finite-difference check: with the live bus the analytic gradient drops the path through the bus by design, and FD
 * would measure that dropped path).
 */
#ifndef CX_CREF_H
#define CX_CREF_H

#include "ref.h"

enum { CREF_BUS_LIVE, CREF_BUS_EXT, CREF_BUS_ZERO };   /* ZERO: the bus stays all zero, as the card with CX_BUS=0 */

typedef struct {
	double hs[CX_DU], xsh, hc[CX_DU];
} cstep;

typedef struct {
	int N, L, bus_mode;
	double *prm;       /* N x CX_NPARAM */
	double *E, *cg;    /* shared */
	double *T;         /* N x CX_NT, the live state */
	rstep *steps;      /* N x L */
	cstep *cs;         /* L */
	double *bus;       /* (L + 1) x [16][128]: bus[t] is the input of step t (the backward reads it again) */
	const double *const *ext;   /* L external bus matrices (CREF_BUS_EXT) */
} cref;

static inline void cref_alloc(cref *C, int N, int L)
{
	C->N = N;
	C->L = L;
	C->prm = calloc((size_t)N * CX_NPARAM, sizeof *C->prm);
	C->E = calloc((size_t)CX_VOCAB * CX_DE, sizeof *C->E);
	C->cg = calloc(CX_DU, sizeof *C->cg);
	C->T = calloc((size_t)N * CX_NT, sizeof *C->T);
	C->steps = calloc((size_t)N * L, sizeof *C->steps);
	C->cs = calloc((size_t)L, sizeof *C->cs);
	C->bus = calloc((size_t)(L + 1) * CX_DB * CX_NBUS, sizeof *C->bus);
	C->bus_mode = CREF_BUS_LIVE;
}

static inline void cref_init(cref *C)
{
	rparams P = { C->prm, C->E, C->cg };
	rparams_init_shared(&P);
	for (int c = 0; c < C->N; c++) rparams_init_col(C->prm + (long)c * CX_NPARAM, c);
}

typedef struct { double *prm, *E, *cg; } cgrads;   /* prm: N x CX_NPARAM */

static inline void cgrads_alloc(cgrads *G, int N)
{
	G->prm = calloc((size_t)N * CX_NPARAM, sizeof *G->prm);
	G->E = calloc((size_t)CX_VOCAB * CX_DE, sizeof *G->E);
	G->cg = calloc(CX_DU, sizeof *G->cg);
}

static inline void cgrads_zero(cgrads *G, int N)
{
	memset(G->prm, 0, (size_t)N * CX_NPARAM * sizeof *G->prm);
	memset(G->E, 0, (size_t)CX_VOCAB * CX_DE * sizeof *G->E);
	memset(G->cg, 0, CX_DU * sizeof *G->cg);
}

/* one cluster step t (forward of every column, vote, consensus, next bus) */
static inline void cref_step(cref *C, int t, uint32_t id)
{
	double *bt = C->bus + (long)t * CX_DB * CX_NBUS, *bn = bt + CX_DB * CX_NBUS;
	static const double zero_bus[CX_DB * CX_NBUS] = { 0 };
	const double *Bt = C->bus_mode == CREF_BUS_EXT ? C->ext[t] : C->bus_mode == CREF_BUS_ZERO ? zero_bus : bt;
	cstep *s = &C->cs[t];
	for (int i = 0; i < CX_DU; i++) s->hs[i] = 0;
	for (int c = 0; c < C->N; c++) {
		rstep *r = &C->steps[(long)c * C->L + t];
		ref_col_forward(C->prm + (long)c * CX_NPARAM, C->E, C->T + (long)c * CX_NT, r, id, Bt);
		for (int i = 0; i < CX_DU; i++) s->hs[i] += r->g * r->u[i];
		for (int d = 0; d < CX_DB; d++) bn[d * CX_NBUS + c] = r->bc[d];
	}
	double ss = 0;
	for (int i = 0; i < CX_DU; i++) ss += s->hs[i] * s->hs[i];
	s->xsh = 1.0 / sqrt(ss / CX_DU + (double)CX_EPS);
	for (int i = 0; i < CX_DU; i++) s->hc[i] = s->hs[i] * s->xsh * C->cg[i];
}

/* backward of step t given dl (scaled); dT: N x CX_NT carried */
static inline void cref_backward(cref *C, cgrads *G, int t, const double *dl, double *dT)
{
	cstep *s = &C->cs[t];
	double dhc[CX_DU] = { 0 }, dhs[CX_DU];
	ref_head_backward(C->E, G->E, s->hc, dl, dhc);
	ref_norm_backward(dhc, s->hs, s->xsh, C->cg, G->cg, dhs, CX_DU);
	for (int c = 0; c < C->N; c++) {
		rstep *r = &C->steps[(long)c * C->L + t];
		double du[CX_DU], dg = 0;
		for (int i = 0; i < CX_DU; i++) { du[i] = r->g * dhs[i]; dg += r->u[i] * dhs[i]; }
		ref_col_backward(C->prm + (long)c * CX_NPARAM, G->prm + (long)c * CX_NPARAM, C->E, G->E, r, du, dg * r->g * (1.0 - r->g), dT + (long)c * CX_NT);
	}
}

/* a chunk from the given state (T and the step-0 bus copied in, not modified): mean loss; gradients if G. After it, bus[L] is
 * the next chunk's step-0 bus. */
static inline double cref_chunk(cref *C, cgrads *G, const double *T0, const double *bus0, const uint32_t *ids, int L)
{
	memcpy(C->T, T0, (size_t)C->N * CX_NT * sizeof *C->T);
	memcpy(C->bus, bus0, CX_DB * CX_NBUS * sizeof *C->bus);
	double *dl = malloc(CX_VOCAB * sizeof *dl), loss = 0;
	double *dls = G ? malloc((size_t)L * CX_VOCAB * sizeof *dls) : NULL;
	for (int t = 0; t < L; t++) cref_step(C, t, ids[t]);
	for (int t = 0; t < L; t++) {
		loss += ref_head_hc(C->E, C->cs[t].hc, ids[t + 1], dl);
		if (G) for (int v = 0; v < CX_VOCAB; v++) dls[(long)t * CX_VOCAB + v] = dl[v] / L;
	}
	if (G) {
		double *dT = calloc((size_t)C->N * CX_NT, sizeof *dT);
		for (int t = L - 1; t >= 0; t--) cref_backward(C, G, t, dls + (long)t * CX_VOCAB, dT);
		free(dT);
		free(dls);
	}
	free(dl);
	return loss / L;
}

#endif
