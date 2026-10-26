/*
 * cortexd: the Intel Phi Cortex generation daemon on the card (Stage 5). usage: cortexd [PART_MIB]
 *
 * 120 column threads (card/cluster.h); thread 0 also serves the host's mailbox (lib/cx-proto.h, part at 1344 MiB of
 * the shared host-memory region, mapped uncached through /dev/phihost) and drives DMA channel 2 (card/cdma.h):
 *   LOAD   the host staged a checkpoint in the part; the card pulls 120 column weight blocks, cg and the embedding
 *          into place by DMA (each thread's huge page has a known physical address), then rebuilds the shadow ES.
 *   RESET  every column's fast weights T and the bus to zero.
 *   BUS    the bus on or off (as the checkpoint was trained).
 *   STEP   one cluster forward on the given token; the 4096 raw logits are assembled in a card huge page and pushed
 *          to the host part by DMA, the ACK word being the DMA's status (written after the data).
 * Everything the host does (sampling, detokenising) runs on the 5800X while the card waits for the next token.
 */
#include "cluster.h"
#include "cdma.h"
#include "../lib/cx-proto.h"

#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

static cluster C __attribute__((aligned(64)));
static volatile uint8_t *part;
static cdma dma;
static double *logits_buf;
static uint64_t logits_phys, cg_phys, etb_phys[1], prm_phys[CL_N];
static uint64_t part_mib = CX_PART_MIB;

static inline uint32_t rd32(int o) { return *(volatile uint32_t *)(part + o); }
static inline uint64_t rd64(int o) { return *(volatile uint64_t *)(part + o); }
static inline void wr32(int o, uint32_t v) { *(volatile uint32_t *)(part + o) = v; }
static inline void wr64(int o, uint64_t v) { *(volatile uint64_t *)(part + o) = v; }

static uint64_t phys_of(const void *p)
{
	int fd = open("/proc/self/pagemap", O_RDONLY);
	if (fd < 0) return 0;
	uint64_t e = 0;
	const off_t at = (off_t)((uintptr_t)p / 4096) * 8;
	const ssize_t r = pread(fd, &e, 8, at);
	close(fd);
	if (r != 8 || !(e >> 63)) return 0;
	return (e & ((1ULL << 55) - 1)) * 4096 + ((uintptr_t)p & 4095);
}

static void *huge(size_t bytes)
{
	void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE, -1, 0);
	if (p == MAP_FAILED) { perror("huge pages"); exit(1); }
	mlock(p, bytes);
	return p;
}

/* per-thread arenas (as tools/cluster-gate.c setup), plus the physical address of the weights for the loader */
static void setup(void *arg, int k)
{
	(void)arg;
	char *a = huge(8u << 20);
	cx_train *tr = &C.tr[k];
	memset(tr, 0, sizeof *tr);
	tr->L = CX_LMAX;
	const size_t hot = (CX_NPARAM + CX_NT + CX_NACT) * 8, lines = ((2u << 20) - hot) / 64, off = ((size_t)k * 104729u % lines) * 64;
	tr->col.prm = (double *)(a + off);
	tr->col.T = tr->col.prm + CX_NPARAM;
	tr->col.act = tr->col.T + CX_NT;
	size_t o = 2u << 20;
#define TAKE(n) (o += ((n) + 63) & ~(size_t)63, (double *)(a + o - (((n) + 63) & ~(size_t)63)))
	tr->acts = TAKE((size_t)CX_LMAX * CX_NACT * 8);
	tr->Ts = TAKE((size_t)CX_LMAX * CX_NT * 8);
	tr->hc = TAKE((size_t)CX_LMAX * CX_DU * 8);
	tr->dhc = TAKE((size_t)CX_LMAX * CX_DU * 8);
	tr->loss = TAKE(CX_LMAX * 8);
	tr->ebuf = TAKE(CX_DE * 8);
	tr->dT = TAKE(CX_NT * 8);
	tr->gsave = TAKE((size_t)CX_LMAX * CX_NG * 8);
	tr->gprm = TAKE(3 * ((CX_NPARAM + 7) & ~7) * 8);
	tr->gcg = TAKE(CX_DU * 8);
	tr->gb1 = TAKE(CX_DH * 8);
	tr->ggain = TAKE(CX_DX * 8);
	tr->gwg = TAKE(CX_DH * 8);
	tr->gbg = TAKE(16 * 8);
	tr->tmp = TAKE(1024 * 8);
	C.hs[k] = TAKE((size_t)CX_LMAX * CX_DU * 8);
	C.xsh[k] = TAKE((size_t)CX_LMAX * 8 * 8);
	C.hcp[k] = TAKE((size_t)CX_LMAX * CX_DU * 8);
	C.dhc[k] = TAKE((size_t)CX_LMAX * CX_DU * 8);
	C.lg[k] = TAKE((size_t)CX_LMAX * CL_MAXBLK * 16 * 8);
	C.tgtrow[k] = TAKE(CX_DE * 8);
	if (o > (8u << 20)) { fprintf(stderr, "cortexd: arena overflow %zu\n", o); exit(1); }
	tr->ET = C.ETB;
	tr->ES = C.ES;
	tr->GET = C.GET;
	tr->cg = C.cg;
	cx_col_init(tr->col.prm, k);
	memset(tr->col.T, 0, CX_NT * 8);
	memset(tr->col.act, 0, CX_NACT * 8);
	cl_blocks(&C, k);
	C.epoch[k] = 0;
	prm_phys[k] = phys_of(tr->col.prm);
}

/* thread 0: pull the staged checkpoint into place by DMA */
static int load_ckpt(void)
{
	const uint64_t src0 = dma.region_addr + (part_mib << 20) + CX_STAGE;
	uint64_t src = src0, seq = 0;
	for (int k = 0; k < CL_N; k++) {
		if (!prm_phys[k]) { fprintf(stderr, "cortexd: no physical address for column %d\n", k); return -1; }
		seq = cdma_submit(&dma, src, prm_phys[k], CX_NPARAM * 8, 0, 0);
		if (!seq) return -1;
		src += CX_NPARAM * 8;
	}
	seq = cdma_submit(&dma, src, cg_phys, CX_DU * 8, 0, 0);
	src += CX_DU * 8;
	for (uint32_t done = 0; done < (4u << 20); done += (1u << 20) - 64) {
		const uint32_t n = (4u << 20) - done < (1u << 20) - 64 ? (4u << 20) - done : (1u << 20) - 64;
		seq = cdma_submit(&dma, src + done, etb_phys[0] + done, n, 0, 0);
		if (!seq) return -1;
	}
	if (cdma_wait(&dma, seq, 10.0)) { fprintf(stderr, "cortexd: checkpoint DMA timed out\n"); return -1; }
	for (int v = 0; v < CX_VOCAB; v++) for (int c = 0; c < CX_DE; c++) C.ES[(long)v * CX_DE + c] = C.ETB[CX_ETB(v, c)];
	return 0;
}

static volatile uint32_t ctl_op, ctl_arg, ctl_quit, ctl_exit;
static int step_t;   /* the step slot within the chunk buffers (0..L-1); the bus wraps with the chunk */

static void serve(void *arg, int k)
{
	(void)arg;
	uint64_t done = 0;
	if (!k) {
		done = CX_CMD_SEQ(rd64(CX_H_CMD));
		wr64(CX_C_ACK, done);
		cx_fence();
		wr64(CX_C_SESSION, rd64(CX_H_SESSION));
		wr32(CX_C_STATE, 1);
	}
	for (;;) {
		/* thread 0 waits for a command and publishes it; everyone else waits at the barrier */
		if (!k) {
			for (;;) {
				const uint64_t c = rd64(CX_H_CMD), seq = CX_CMD_SEQ(c);
				if (seq == CX_CMD_QUIT) { ctl_quit = 1; break; }
				if (seq != done) { ctl_op = rd32(CX_H_CMD_OP); ctl_arg = CX_CMD_ARG(c); done = seq; break; }
				__asm__ volatile("delay %0" : : "r"(16u));
			}
		}
		cl_wait(&C, k);
		if (ctl_quit) break;
		const uint32_t op = ctl_op, a = ctl_arg;
		if (op == CX_OP_LOAD) {
			if (!k) { if (load_ckpt()) wr32(CX_C_STATE, 2); }
			cl_wait(&C, k);
		} else if (op == CX_OP_RESET) {
			memset(C.tr[k].col.T, 0, CX_NT * 8);
			if (!k) { memset(C.bus, 0, (size_t)(CX_LMAX + 1) * CL_BUS * 8); step_t = 0; }
			cl_wait(&C, k);
		} else if (op == CX_OP_EXIT) {
			if (!k) { ctl_exit = 1; cx_fence(); wr64(CX_C_ACK, done); }
			break;
		} else if (op == CX_OP_BUS) {
			if (!k) C.bus_on = (int)a;
			cl_wait(&C, k);
		} else if (op == CX_OP_STEP) {
			const int t = step_t;
			cl_fwd_step(&C, k, t, a, a, 0);
			cl_wait(&C, k);   /* every thread's logits are in the buffer */
			if (!k) {
				cx_fence();
				const uint64_t seq = cdma_submit(&dma, logits_phys, dma.region_addr + (part_mib << 20) + CX_LOGITS, CX_VOCAB * 8,
				                                 dma.region_addr + (part_mib << 20) + CX_C_ACK, done);
				if (!seq || cdma_wait(&dma, seq, 2.0)) wr32(CX_C_STATE, 2);
				if (t + 1 == CX_LMAX) { cx_copy(C.bus, C.bus + (long)CX_LMAX * CL_BUS, CL_BUS); step_t = 0; } else step_t = t + 1;
			}
			cl_wait(&C, k);   /* step_t and the wrapped bus visible */
			continue;
		}
		if (!k) { cx_fence(); wr64(CX_C_ACK, done); }
	}
	if (!k) wr32(CX_C_STATE, 3);
}

int main(int argc, char **argv)
{
	if (argc > 1) part_mib = strtoull(argv[1], NULL, 10);
	C.L = CX_LMAX;
	C.bus_on = 1;
	C.ETB = huge(4u << 20);
	etb_phys[0] = phys_of(C.ETB);
	C.GET = huge(4u << 20);
	C.m_et = C.GET;   /* unused in the daemon */
	C.v_et = C.GET;
	C.ES = huge(4u << 20);
	char *sh = huge(16u << 20);
	size_t o = 0;
#define STAKE(n) (o += ((n) + 63) & ~(size_t)63, (double *)(sh + o - (((n) + 63) & ~(size_t)63)))
	C.cg = STAKE(CX_DU * 8);
	C.m_cg = STAKE(CX_DU * 8);
	C.v_cg = STAKE(CX_DU * 8);
	C.bus = STAKE((size_t)(CX_LMAX + 1) * CL_BUS * 8);
	C.votes = STAKE(2 * CL_N * CX_DU * 8);
	C.gsum = STAKE(2 * CL_GROUPS * CX_DU * 8);
	C.pairs = STAKE(2 * CL_N * 8 * 8);
	C.dhcp = STAKE((size_t)CL_N * CX_LMAX * CX_DU * 8);
	C.gdhc = STAKE((size_t)CL_GROUPS * CX_LMAX * CX_DU * 8);
	C.dep = STAKE((size_t)CL_N * CX_LMAX * CX_DU * 8);
	C.loss = STAKE(CX_LMAX * 8);
	C.argmax = (volatile uint32_t *)STAKE(2 * 16 * 8);
	logits_buf = STAKE(CX_VOCAB * 8);
	if (o > (16u << 20)) { fprintf(stderr, "cortexd: shared overflow %zu\n", o); return 1; }
	memset(sh, 0, o);
	C.logits_out = logits_buf;
	cg_phys = phys_of(C.cg);
	logits_phys = phys_of(logits_buf);
	if (!etb_phys[0] || !cg_phys || !logits_phys) { fprintf(stderr, "cortexd: physical addresses unavailable\n"); return 1; }
	for (int i = 0; i < CX_DU; i++) C.cg[i] = 1.0;

	if (cdma_open(&dma, 2)) return 1;
	int fd = open("/dev/phihost", O_RDWR | O_SYNC);
	part = fd < 0 ? MAP_FAILED : mmap(NULL, CX_STAGE + (48u << 20), PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)(part_mib << 20));
	if (part == MAP_FAILED) { perror("/dev/phihost"); return 1; }
	wr32(CX_C_STATE, 0);

	cx_barrier_init(&C.bar, CL_N, 2, CX_CORE_REL);
	cx_team_run(CL_N, setup, NULL);
	printf("cortexd: %d columns, part %llu MiB, DMA channel 2 (region 0x%llx, %llu MiB); waiting for a session\n", CL_N,
	       (unsigned long long)part_mib, (unsigned long long)dma.region_addr, (unsigned long long)(dma.region_size >> 20));
	fflush(stdout);
	for (;;) {
		const uint64_t session = rd64(CX_C_SESSION);
		while (rd32(CX_H_MAGIC) != CX_MAGIC || rd64(CX_H_SESSION) == session) usleep(1000);
		ctl_quit = 0;
		cx_team_run(CL_N, serve, NULL);
		printf("cortexd: session %llu ended%s\n", (unsigned long long)rd64(CX_C_SESSION), ctl_exit ? ", exiting" : "; waiting for the next");
		fflush(stdout);
		if (ctl_exit) break;
	}
	return 0;
}
