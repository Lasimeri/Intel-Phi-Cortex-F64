/*
 * hello: Stage 0b. Proves the card toolchain and the run path: prints the CPU it runs on, one VPU add, the TSC rate
 * against CLOCK_MONOTONIC (1 s), and the SBOX DMA control register (DCR) read-only through /dev/mem, decoded per
 * channel (bit 2n+1 enabled, bit 2n host owned; both 0 = card owned and disabled, cdma.md).
 */
#include "knc.h"

#include <fcntl.h>
#include <immintrin.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>

#define SBOX_DMA 0x08007DA000ull   /* SBOX + 0xA000: the channels' registers */
#define R_DCR 0x280

int main(void)
{
	printf("cpu %d\n", sched_getcpu());

	float out[16] __attribute__((aligned(64)));
	_mm512_store_ps(out, _mm512_add_ps(_mm512_set1_ps(1.5f), _mm512_set1_ps(2.25f)));
	printf("vpu add 1.5 + 2.25 = %g (lane 0), %g (lane 15)\n", out[0], out[15]);

	const double t0 = cx_now();
	const uint64_t c0 = cx_rdtsc();
	while (cx_now() - t0 < 1.0) {}
	const uint64_t c1 = cx_rdtsc();
	const double t1 = cx_now();
	printf("tsc %.3f MHz over %.6f s\n", (double)(c1 - c0) / (t1 - t0) / 1e6, t1 - t0);

	int fd = open("/dev/mem", O_RDONLY | O_SYNC);
	volatile uint32_t *r = fd < 0 ? MAP_FAILED : mmap(NULL, 4096, PROT_READ, MAP_SHARED, fd, (off_t)SBOX_DMA);
	if (r == MAP_FAILED) { perror("/dev/mem"); return 1; }
	const uint32_t dcr = r[R_DCR / 4];
	printf("dcr 0x%08x:", dcr);
	for (int ch = 0; ch < 8; ch++)
		printf(" ch%d=%s%s", ch, dcr >> (2 * ch + 1) & 1 ? "on" : "off", dcr >> (2 * ch) & 1 ? "/host" : "/card");
	printf("\n");
	return 0;
}
