/*
 * knc.h: card-side helpers for Cortex's card programs (S3 clang -march=knc). Header only.
 *
 * Thread map (PLAN.md "Card process layout"): column k (0..119) runs on core k/2. Cores 0-58 use hardware threads 0
 * and 1 (CPU 1 + 4 * core + thread). Core 59 uses threads 0 and 3 (CPUs 237 and 240), beside the block pollers on
 * CPUs 238 and 239. The dispatcher runs on CPU 241 (core 60, thread 1). Core 60 thread 0 is CPU 0, the OS.
 */
#ifndef CX_KNC_H
#define CX_KNC_H

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <time.h>

#define CX_COLUMNS 120
#define CX_DISPATCH_CPU 241

static inline int cx_cpu(int k)
{
	const int core = k / 2, t = k % 2;
	if (core == 59) return t ? 240 : 237;
	return 1 + 4 * core + t;
}

static inline int cx_pin(int cpu)
{
	cpu_set_t s;
	CPU_ZERO(&s);
	CPU_SET(cpu, &s);
	return pthread_setaffinity_np(pthread_self(), sizeof s, &s);
}

static inline uint64_t cx_rdtsc(void)
{
	uint32_t lo, hi;
	__asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
	return (uint64_t)hi << 32 | lo;
}

/* one spin-wait pause: KNC DELAY gives the core's other hardware threads the issue slots (pattern from the user's
 * earlier KNC thread pool, lib/pool.c relax()) */
static inline void cx_relax(void) { __asm__ volatile("delay %0" : : "r"(16u)); }

/* the card has no mfence; a locked read-modify-write is the full fence */
static inline void cx_fence(void) { __asm__ volatile("lock addq $0, (%%rsp)" ::: "memory"); }

static inline double cx_now(void)
{
	struct timespec s;
	clock_gettime(CLOCK_MONOTONIC, &s);
	return s.tv_sec + s.tv_nsec / 1e9;
}

#endif
