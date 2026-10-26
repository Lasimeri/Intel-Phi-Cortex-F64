/*
 * mbox-ping: Stage 0e, host side. usage: mbox-ping OUTDIR [ROUNDS] [REPEATS]
 *
 * Maps Cortex's part of /dev/shm/phi-hostmem (lib/cx-proto.h), opens a session with the card's mbox-gate, then for
 * each repeat runs ROUNDS round trips per payload (0 and 256 lines = 16 KiB, interleaved): write command n, spin on
 * the card's ACK, check the payload's first and last words. Round-trip ns go to OUTDIR/mbox-PAYLOAD-rREPEAT.u32
 * (build/host/stats with TICK_MHZ 1000). Pinned to host CPU 2. Ends with CX_CMD_QUIT.
 */
#define _GNU_SOURCE
#include "../lib/cx-proto.h"

#include <fcntl.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

static uint8_t *part;
#define LD64(o) __atomic_load_n((uint64_t *)(part + (o)), __ATOMIC_ACQUIRE)
#define LD32(o) __atomic_load_n((uint32_t *)(part + (o)), __ATOMIC_ACQUIRE)
#define ST64(o, x) __atomic_store_n((uint64_t *)(part + (o)), (x), __ATOMIC_RELEASE)
#define ST32(o, x) __atomic_store_n((uint32_t *)(part + (o)), (x), __ATOMIC_RELEASE)

static uint64_t ns(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return (uint64_t)t.tv_sec * 1000000000u + (uint64_t)t.tv_nsec;
}

int main(int argc, char **argv)
{
	if (argc < 2) { fprintf(stderr, "usage: mbox-ping OUTDIR [ROUNDS] [REPEATS]\n"); return 2; }
	const long rounds = argc > 2 ? atol(argv[2]) : 10000;
	const int repeats = argc > 3 ? atoi(argv[3]) : 5;
	cpu_set_t set;
	CPU_ZERO(&set);
	CPU_SET(2, &set);
	sched_setaffinity(0, sizeof set, &set);

	int fd = open("/dev/shm/phi-hostmem", O_RDWR);
	if (fd < 0) { perror("/dev/shm/phi-hostmem"); return 1; }
	part = mmap(NULL, CX_PART_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)CX_PART_MIB << 20);
	close(fd);
	if (part == MAP_FAILED) { perror("mmap"); return 1; }

	ST32(CX_H_MAGIC, CX_MAGIC);
	ST32(CX_H_VERSION, CX_VERSION);
	uint64_t seq = CX_CMD_SEQ(LD64(CX_H_CMD));
	if (seq == CX_CMD_QUIT) seq = 0;
	ST64(CX_H_CMD, CX_CMD(seq, 0));
	const uint64_t session = ns();
	ST64(CX_H_SESSION, session);
	const uint64_t t0 = ns();
	while (LD64(CX_C_SESSION) != session || LD32(CX_C_STATE) != 1) {
		if (ns() - t0 > 30000000000ull) { fprintf(stderr, "mbox-ping: the card did not take the session in 30 s\n"); return 1; }
		usleep(1000);
	}
	printf("session %llu taken\n", (unsigned long long)session);

	uint32_t *rtt = malloc((size_t)rounds * sizeof *rtt);
	const uint32_t payload[2] = { 0, 256 };
	long bad = 0;
	for (int rep = 0; rep < repeats; rep++)
		for (int p = 0; p < 2; p++) {
			const uint32_t lines = payload[p];
			for (long r = 0; r < rounds; r++) {
				seq++;
				const uint64_t a = ns();
				ST64(CX_H_CMD, CX_CMD(seq, lines));
				while (LD64(CX_C_ACK) != seq) {}
				const uint64_t b = ns();
				rtt[r] = (uint32_t)(b - a);
				if (lines) {
					const uint32_t *d = (const uint32_t *)(part + CX_DATA);
					const uint32_t first = (uint32_t)(seq * 65536), last = (uint32_t)(seq * 65536 + (lines - 1) * 16 + 15);
					bad += d[0] != first || d[lines * 16 - 1] != last;
				}
			}
			char path[4096];
			snprintf(path, sizeof path, "%s/mbox-%u-r%d.u32", argv[1], lines * 64, rep);
			FILE *f = fopen(path, "wb");
			if (!f || fwrite(rtt, sizeof *rtt, (size_t)rounds, f) != (size_t)rounds || fclose(f)) { perror(path); return 1; }
		}
	ST64(CX_H_CMD, CX_CMD(CX_CMD_QUIT, 0));
	printf("done: %d repeats x 2 payloads x %ld rounds; payload check mismatches %ld\n", repeats, rounds, bad);
	return 0;
}
