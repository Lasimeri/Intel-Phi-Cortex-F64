/*
 * mbox-gate: Stage 0e, card side (the dispatcher's mailbox path). usage: mbox-gate [PART_MIB] [DELAY]
 *
 * Pinned to CPU 241 (knc.h). Maps the part's first CX_DATA + 64 KiB of the region uncached through /dev/phihost,
 * takes the host's session, then answers commands: for command n with argument L (payload in 64-byte lines) it
 * writes L lines of pattern (word i of line j = n * 65536 + j * 16 + i) into CX_DATA with vector stores, then ACK = n.
 * Polls the host's command word with DELAY (KNC delay cycles, 0 = none) between reads. Stops on CX_CMD_QUIT.
 */
#include "../card/knc.h"
#include "../lib/cx-proto.h"

#include <fcntl.h>
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

static volatile uint8_t *part;
static inline uint32_t rd32(int o) { return *(volatile uint32_t *)(part + o); }
static inline uint64_t rd64(int o) { return *(volatile uint64_t *)(part + o); }
static inline void wr32(int o, uint32_t v) { *(volatile uint32_t *)(part + o) = v; }
static inline void wr64(int o, uint64_t v) { *(volatile uint64_t *)(part + o) = v; }

int main(int argc, char **argv)
{
	const uint64_t part_mib = argc > 1 ? strtoull(argv[1], NULL, 10) : CX_PART_MIB;
	const uint32_t dly = argc > 2 ? (uint32_t)atoi(argv[2]) : 16;
	cx_pin(CX_DISPATCH_CPU);
	const size_t len = CX_DATA + (64u << 10);
	int fd = open("/dev/phihost", O_RDWR | O_SYNC);
	part = fd < 0 ? MAP_FAILED : mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)(part_mib << 20));
	if (part == MAP_FAILED) { perror("/dev/phihost"); return 1; }
	wr32(CX_C_STATE, 0);
	printf("mbox-gate: cpu %d, part %llu MiB, delay %u; waiting for a session\n", sched_getcpu(), (unsigned long long)part_mib, dly);
	fflush(stdout);

	uint64_t session = rd64(CX_C_SESSION), done = 0, heart = 0;
	const double t0 = cx_now();
	while (rd32(CX_H_MAGIC) != CX_MAGIC || rd64(CX_H_SESSION) == session) {
		if (cx_now() - t0 > 60) { printf("mbox-gate: no session in 60 s\n"); return 1; }
		usleep(1000);
	}
	session = rd64(CX_H_SESSION);
	done = CX_CMD_SEQ(rd64(CX_H_CMD));   /* never replay a command from before this session */
	wr64(CX_C_ACK, done);
	cx_fence();
	wr64(CX_C_SESSION, session);
	wr32(CX_C_STATE, 1);
	printf("mbox-gate: session %llu taken at seq %llu\n", (unsigned long long)session, (unsigned long long)done);
	fflush(stdout);

	static const int idx[16] __attribute__((aligned(64))) = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
	const __m512i lane = _mm512_load_epi32(idx);   /* the KNC header has no _mm512_set_epi32 */
	long served = 0;
	for (;;) {
		const uint64_t c = rd64(CX_H_CMD), seq = CX_CMD_SEQ(c);
		if (seq == CX_CMD_QUIT) break;
		if (seq == done) {
			if (dly) __asm__ volatile("delay %0" : : "r"(dly));
			continue;
		}
		const uint32_t lines = CX_CMD_ARG(c);
		for (uint32_t j = 0; j < lines; j++) {
			const __m512i base = _mm512_set1_epi32((int)(seq * 65536 + j * 16));
			_mm512_store_epi32((void *)(part + CX_DATA + 64 * j), _mm512_add_epi32(base, lane));
		}
		cx_fence();
		wr64(CX_C_ACK, seq);
		wr64(CX_C_HEART, ++heart);
		done = seq;
		served++;
	}
	wr32(CX_C_STATE, 3);
	printf("mbox-gate: served %ld commands in %.3f s\n", served, cx_now() - t0);
	return 0;
}
