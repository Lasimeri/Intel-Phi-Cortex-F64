/*
 * team.h: the 120 column threads (knc.h thread map) and their barriers. Header only.
 *
 * cx_team_run(n, fn, arg) runs fn(arg, k) for k = 0..n-1, each on its own pinned thread (k = 0 on the caller, which
 * is pinned too), and returns when all have returned.
 *
 * Barriers, each with one 64-byte line per flag so no two threads write the same line except where stated:
 *   central - one shared counter (locked add by every thread) and one release epoch;
 *   tree    - arrival up a static tree of arity A (node k's children are k*A+1 .. k*A+A): a node waits for its
 *             children's arrival epochs, then posts its own; the root (k = 0) writes the release epoch, which every
 *             thread spins on. A = n-1 is a flat flag array polled by thread 0.
 */
#ifndef CX_TEAM_H
#define CX_TEAM_H

#include "knc.h"

#include <stdlib.h>

typedef struct { volatile uint64_t v; char pad[56]; } cx_line;

/* shapes: CX_CENTRAL counter; CX_TREE arrival tree + one release line; CX_TREE_REL release travels back down the
 * same tree (each release line has one reader); CX_CORE_REL like CX_TREE_REL over a core-aware tree: an odd thread
 * hangs off its core partner (k - 1, same core: shared L1/L2), even threads (core leaders) form the arity tree over
 * cores */
enum { CX_CENTRAL, CX_TREE, CX_TREE_REL, CX_CORE_REL };

typedef struct {
	int n, arity, shape;
	cx_line count, release;
	cx_line arrive[CX_COLUMNS];
	cx_line rel[CX_COLUMNS];
	int nkids[CX_COLUMNS];
	int kids[CX_COLUMNS][9];
} cx_barrier;

static inline void cx_barrier_init(cx_barrier *b, int n, int arity, int shape)
{
	b->n = n;
	b->arity = arity;
	b->shape = shape;
	b->count.v = 0;
	b->release.v = 0;
	for (int k = 0; k < CX_COLUMNS; k++) {
		b->arrive[k].v = 0;
		b->rel[k].v = 0;
		b->nkids[k] = 0;
	}
	for (int k = 0; k < n; k++) {
		if (shape == CX_CORE_REL) {
			if (k % 2) continue;
			const int core = k / 2;
			if (k + 1 < n) b->kids[k][b->nkids[k]++] = k + 1;
			for (int j = 1; j <= arity && core * arity + j < (n + 1) / 2; j++) b->kids[k][b->nkids[k]++] = 2 * (core * arity + j);
		} else {
			for (int c = k * arity + 1; c <= k * arity + arity && c < n && b->nkids[k] < 9; c++) b->kids[k][b->nkids[k]++] = c;
		}
	}
}

/* thread k's barrier number epoch (1, 2, 3, ... per thread, the same sequence on every thread) */
static inline void cx_barrier_wait(cx_barrier *b, int k, uint64_t epoch)
{
	if (b->shape == CX_CENTRAL) {
		if (__atomic_add_fetch(&b->count.v, 1, __ATOMIC_ACQ_REL) == (uint64_t)b->n * epoch)
			__atomic_store_n(&b->release.v, epoch, __ATOMIC_RELEASE);
		else
			while (__atomic_load_n(&b->release.v, __ATOMIC_ACQUIRE) < epoch) cx_relax();
		return;
	}
	if (b->shape == CX_TREE && b->arity > 8) {   /* flat: thread 0 polls everyone */
		if (k) {
			__atomic_store_n(&b->arrive[k].v, epoch, __ATOMIC_RELEASE);
			while (__atomic_load_n(&b->release.v, __ATOMIC_ACQUIRE) < epoch) cx_relax();
		} else {
			for (int c = 1; c < b->n; c++)
				while (__atomic_load_n(&b->arrive[c].v, __ATOMIC_ACQUIRE) < epoch) cx_relax();
			__atomic_store_n(&b->release.v, epoch, __ATOMIC_RELEASE);
		}
		return;
	}
	for (int i = 0; i < b->nkids[k]; i++)
		while (__atomic_load_n(&b->arrive[b->kids[k][i]].v, __ATOMIC_ACQUIRE) < epoch) cx_relax();
	if (b->shape == CX_TREE) {
		if (k) {
			__atomic_store_n(&b->arrive[k].v, epoch, __ATOMIC_RELEASE);
			while (__atomic_load_n(&b->release.v, __ATOMIC_ACQUIRE) < epoch) cx_relax();
		} else {
			__atomic_store_n(&b->release.v, epoch, __ATOMIC_RELEASE);
		}
		return;
	}
	if (k) {
		__atomic_store_n(&b->arrive[k].v, epoch, __ATOMIC_RELEASE);
		while (__atomic_load_n(&b->rel[k].v, __ATOMIC_ACQUIRE) < epoch) cx_relax();
	}
	for (int i = b->nkids[k] - 1; i >= 0; i--) __atomic_store_n(&b->rel[b->kids[k][i]].v, epoch, __ATOMIC_RELEASE);
}

typedef void (*cx_fn)(void *arg, int k);
typedef struct { cx_fn fn; void *arg; int k; } cx_start;

/* thread k runs on cx_cpu(k * cx_stride): stride 2 = one thread per core (gates only) */
static int cx_stride = 1;
static int cx_cpu0 = -1;   /* CX_CPU0: the CPU of thread 0 when set (single-thread gates on a free core) */
/* the CPU each thread was found on after pinning, and pin failures (pthread_setaffinity_np errors) */
static int cx_found_cpu[CX_COLUMNS];
static int cx_pin_fail;

static void cx_pin_self(int k)
{
	if (cx_pin(k == 0 && cx_cpu0 >= 0 ? cx_cpu0 : cx_cpu(k * cx_stride))) __atomic_fetch_add(&cx_pin_fail, 1, __ATOMIC_RELAXED);
	cx_found_cpu[k] = sched_getcpu();
}

static void *cx_thread(void *v)
{
	cx_start *s = v;
	cx_pin_self(s->k);
	s->fn(s->arg, s->k);
	return NULL;
}

static inline int cx_team_run(int n, cx_fn fn, void *arg)
{
	pthread_t *t = calloc((size_t)n, sizeof *t);
	cx_start *s = calloc((size_t)n, sizeof *s);
	for (int k = 1; k < n; k++) {
		s[k] = (cx_start){ fn, arg, k };
		if (pthread_create(&t[k], NULL, cx_thread, &s[k])) return -1;
	}
	cx_pin_self(0);
	fn(arg, 0);
	for (int k = 1; k < n; k++) pthread_join(t[k], NULL);
	free(t);
	free(s);
	return 0;
}

#endif
