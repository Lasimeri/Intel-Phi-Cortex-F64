/*
 * train-gate: Stage 3, card side. usage: train-gate fwd|bwd TOKENS.u16 OUTDIR L WARM
 *
 * One column (thread pinned to CPU 1 = core 0). Weights, the embedding and the bus ring from lib/cx-col.h (column 0),
 * tokens from TOKENS.u16 (ids[0..WARM+L]). WARM steps from T = 0 without a head, then a chunk of L steps with saves,
 * the head, the losses and dl; `bwd` continues with the backward and dumps every gradient. The host side
 * (tools/train-check.c) replays the same tokens in double.
 *
 * Dumps in OUTDIR: fwd-loss.f32 (L), fwd-dl.f32 (L x 4096), fwd-hc.f32 (L x 128); bwd: gprm.f32 (CX_NPARAM), gcg.f32
 * (128), get.f32 (ETB layout, 4096 x 128), dhc.f32 (L x 128). Prints the time per step of each phase.
 */
#include "../card/team.h"
#include "../card/train.h"

#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <math.h>

static void *huge(size_t bytes)
{
	void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE, -1, 0);
	if (p == MAP_FAILED) { perror("huge pages"); exit(1); }
	return p;
}

static int dump(const char *dir, const char *name, const void *p, size_t bytes)
{
	char path[512];
	snprintf(path, sizeof path, "%s/%s", dir, name);
	FILE *f = fopen(path, "wb");
	if (!f || fwrite(p, 1, bytes, f) != bytes || fclose(f)) { perror(path); return -1; }
	return 0;
}


/* run: train one column on NTOK tokens of TOKENS.u16 in chunks of L with Adam; evaluate on NVAL tokens of VALID.u16.
 * usage: train-gate run TOKENS.u16 OUTDIR L NTOK LR VALID.u16 NVAL [REPORT_EVERY] */
static int run_mode(int argc, char **argv, cx_train *tr, double *ring)
{
	(void)ring;
	if (argc < 9) { fprintf(stderr, "usage: train-gate run TOKENS.u16 OUTDIR L NTOK LR VALID.u16 NVAL [REPORT_EVERY]\n"); return 2; }
	const int L = tr->L;
	const long ntok = atol(argv[5]), nval = atol(argv[8]);
	const double lr = atof(argv[6]);
	const int every = argc > 9 ? atoi(argv[9]) : 100;
	const double b1 = 0.9, b2 = 0.95, eps = 1e-8;
	const int warm_chunks = 50;
	FILE *f = fopen(argv[2], "rb");
	if (!f) { perror(argv[2]); return 1; }
	uint16_t *tok = malloc((size_t)(ntok + 1) * 2);
	const long got = (long)fread(tok, 2, (size_t)ntok + 1, f);
	fclose(f);
	if (got < L + 1) { fprintf(stderr, "train-gate: short token file\n"); return 1; }
	const long nchunks = (got - 1) / L;
	f = fopen(argv[7], "rb");
	if (!f) { perror(argv[7]); return 1; }
	uint16_t *vtok = malloc((size_t)(nval + 1) * 2);
	const long vgot = (long)fread(vtok, 2, (size_t)nval + 1, f);
	fclose(f);

	/* optimizer state */
	double *m_prm = huge(2u << 20), *v_prm = m_prm + ((CX_NPARAM + 7) & ~7);
	double *m_cg = v_prm + ((CX_NPARAM + 7) & ~7), *v_cg = m_cg + CX_DU;
	double *m_et = huge(4u << 20), *v_et = huge(4u << 20);
	cx_zero(m_prm, 2 * ((CX_NPARAM + 7) & ~7) + 2 * CX_DU);
	cx_zero(m_et, (long)CX_VOCAB * CX_DE);
	cx_zero(v_et, (long)CX_VOCAB * CX_DE);
	double *Bt = tr->tmp + 768;   /* 16 x 128 = 2048 floats: beyond tmp; take a dedicated block */
	Bt = aligned_alloc(64, CX_DB * CX_NBUS * 8);
	memset(Bt, 0, CX_DB * CX_NBUS * 8);
	double bc_prev[CX_DB] __attribute__((aligned(64))) = { 0 };
	double *losslog = malloc((size_t)nchunks * 8);
	uint32_t ids[CX_LMAX + 1];

	printf("run: %ld tokens = %ld chunks of %d, lr %g (warm-up %d chunks), b1 %g b2 %g, valid %ld tokens\n", got - 1, nchunks, L, lr, warm_chunks, b1, b2, vgot - 1);
	double b1k = 1, b2k = 1;
	double acc_loss = 0, t_start = cx_now(), t_last = t_start;
	uint64_t cyc[8] = { 0 };
	for (long k = 0; k < nchunks; k++) {
		for (int t = 0; t <= L; t++) ids[t] = tok[k * L + t];
		const uint64_t c0 = cx_rdtsc();
		cx_train_chunk_begin(tr);
		cx_gather_prefetch(tr->ET, ids[0]);
		for (int t = 0; t < L; t++) {
			cx_bus_set0(Bt, t ? tr->acts + (long)(t - 1) * CX_NACT + CX_A_BC : bc_prev);
			cx_train_fwd_step(tr, t, ids[t], ids[t + 1], Bt);
		}
		for (int d = 0; d < CX_DB; d++) bc_prev[d] = tr->acts[(long)(L - 1) * CX_NACT + CX_A_BC + d];
		const uint64_t c1 = cx_rdtsc();
		cx_head_logits(tr);
		cx_head_loss(tr, ids + 1);
		const uint64_t c2 = cx_rdtsc();
		cx_head_dhc(tr);
		const uint64_t c3 = cx_rdtsc();
		for (int t = L - 1; t >= 0; t--) {
			cx_bus_set0(Bt, t ? tr->acts + (long)(t - 1) * CX_NACT + CX_A_BC : bc_prev);   /* bc_prev was overwritten: restore below */
			cx_train_bwd_step(tr, t, Bt);
		}
		const uint64_t c4 = cx_rdtsc();
		cx_train_param_grads(tr);
		cx_zero(tr->GET, (long)CX_VOCAB * CX_DE);
		cx_head_dE(tr);
		cx_train_input_grads(tr, ids);
		const uint64_t c5 = cx_rdtsc();
		/* Adam with bias correction and a linear warm-up */
		b1k *= b1;
		b2k *= b2;
		const double lr_k = lr * (k < warm_chunks ? (double)(k + 1) / warm_chunks : 1.0);
		const double bc2 = sqrt(1.0 - b2k), lr_t = lr_k * bc2 / (1.0 - b1k), eps_t = eps * bc2;
		cx_adam(tr->col.prm, tr->gprm, m_prm, v_prm, (CX_NPARAM + 7) & ~7, b1, b2, lr_t, eps_t);
		cx_adam(tr->cg, tr->gcg, m_cg, v_cg, CX_DU, b1, b2, lr_t, eps_t);
		cx_adam(tr->ET, tr->GET, m_et, v_et, (long)CX_VOCAB * CX_DE, b1, b2, lr_t, eps_t);
		const uint64_t c6 = cx_rdtsc();
		cyc[0] += c1 - c0; cyc[1] += c2 - c1; cyc[2] += c3 - c2; cyc[3] += c4 - c3; cyc[4] += c5 - c4; cyc[5] += c6 - c5;
		double lsum = 0;
		for (int t = 0; t < L; t++) lsum += tr->loss[t];
		losslog[k] = (double)(lsum / L);
		acc_loss += lsum / L;
		if ((k + 1) % every == 0 || k + 1 == nchunks) {
			const double now = cx_now(), n = (k + 1) % every ? (k + 1) % every : every;
			printf("chunk %6ld  tokens %9ld  loss %.4f nats = %.4f bits/token  lr %.2e  %.0f tok/s  (us/step: fwd %.0f head %.0f dhc %.0f bwd %.0f grads %.0f adam %.0f)\n",
			       k + 1, (k + 1) * (long)L, acc_loss / n, acc_loss / n / CX_LN2, lr_k, n * L / (now - t_last),
			       cyc[0] / 1238.093 / (n * L), cyc[1] / 1238.093 / (n * L), cyc[2] / 1238.093 / (n * L), cyc[3] / 1238.093 / (n * L), cyc[4] / 1238.093 / (n * L), cyc[5] / 1238.093 / (n * L));
			fflush(stdout);
			acc_loss = 0;
			t_last = now;
			memset(cyc, 0, sizeof cyc);
		}
	}
	printf("run: %ld chunks in %.1f s\n", nchunks, cx_now() - t_start);
	if (dump(argv[3], "run-loss.f32", losslog, (size_t)nchunks * 8)) return 1;

	/* evaluation: forward + head only, T from zero, own bus continuing */
	const long vchunks = (vgot - 1) / L;
	memset(tr->col.T, 0, CX_NT * 8);
	memset(bc_prev, 0, sizeof bc_prev);
	double vsum = 0;
	for (long k = 0; k < vchunks; k++) {
		for (int t = 0; t <= L; t++) ids[t] = vtok[k * L + t];
		cx_gather_prefetch(tr->ET, ids[0]);
		for (int t = 0; t < L; t++) {
			cx_bus_set0(Bt, t ? tr->acts + (long)(t - 1) * CX_NACT + CX_A_BC : bc_prev);
			cx_train_fwd_step(tr, t, ids[t], ids[t + 1], Bt);
		}
		for (int d = 0; d < CX_DB; d++) bc_prev[d] = tr->acts[(long)(L - 1) * CX_NACT + CX_A_BC + d];
		cx_head_logits(tr);
		cx_head_loss(tr, ids + 1);
		for (int t = 0; t < L; t++) vsum += tr->loss[t];
	}
	printf("valid: %ld tokens, loss %.4f nats = %.4f bits/token = %.4f bits/byte (3.961 B/token)\n", vchunks * L, vsum / (vchunks * L), vsum / (vchunks * L) / CX_LN2,
	       vsum / (vchunks * L) / CX_LN2 / 3.961);
	/* checkpoint: prm, cg, ETB */
	char path[512];
	snprintf(path, sizeof path, "%s/ckpt.bin", argv[3]);
	f = fopen(path, "wb");
	if (!f || fwrite(tr->col.prm, 8, CX_NPARAM, f) != CX_NPARAM || fwrite(tr->cg, 8, CX_DU, f) != CX_DU ||
	    fwrite(tr->ET, 8, (size_t)CX_VOCAB * CX_DE, f) != (size_t)CX_VOCAB * CX_DE || fclose(f)) { perror(path); return 1; }
	return 0;
}

#define US(a, b) ((b - a) / 1238.093)

int main(int argc, char **argv)
{
	if (argc < 6 || (strcmp(argv[1], "fwd") && strcmp(argv[1], "bwd") && strcmp(argv[1], "run"))) { fprintf(stderr, "usage: train-gate fwd|bwd TOKENS.u16 OUTDIR L WARM | run TOKENS.u16 OUTDIR L NTOK LR VALID.u16 NVAL [EVERY]\n"); return 2; }
	const int bwd = !strcmp(argv[1], "bwd"), runm = !strcmp(argv[1], "run"), L = atoi(argv[4]), warm = runm ? 0 : atoi(argv[5]);
	if (L < 16 || L > CX_LMAX || L % 16) { fprintf(stderr, "train-gate: L must be 16 or 32\n"); return 2; }
	cx_pin(getenv("CX_CPU") ? atoi(getenv("CX_CPU")) : cx_cpu(0));

	char *a = huge(8u << 20);
	size_t off = 0;
#define TAKE(n) (off += ((n) + 63) & ~(size_t)63, (double *)(a + off - (((n) + 63) & ~(size_t)63)))
	cx_train tr;
	memset(&tr, 0, sizeof tr);
	tr.L = L;
	tr.col.prm = TAKE(CX_NPARAM * 8);
	tr.col.T = TAKE(CX_NT * 8);
	tr.col.act = TAKE(CX_NACT * 8);
	tr.acts = TAKE((size_t)CX_LMAX * CX_NACT * 8);
	tr.Ts = TAKE((size_t)CX_LMAX * CX_NT * 8);
	tr.logits = TAKE((size_t)CX_LMAX * CX_VOCAB * 8);
	tr.hc = TAKE((size_t)CX_LMAX * CX_DU * 8);
	tr.dhc = TAKE((size_t)CX_LMAX * CX_DU * 8);
	tr.cg = TAKE(CX_DU * 8);
	tr.loss = TAKE(CX_LMAX * 8);
	tr.ebuf = TAKE(CX_DE * 8);
	tr.dT = TAKE(CX_NT * 8);
	tr.gsave = TAKE((size_t)CX_LMAX * CX_NG * 8);
	tr.gprm = TAKE(CX_NPARAM * 8);
	tr.gcg = TAKE(CX_DU * 8);
	tr.gb1 = TAKE(CX_DH * 8);
	tr.ggain = TAKE(CX_DX * 8);
	tr.gwg = TAKE(CX_DH * 8);
	tr.gbg = TAKE(16 * 8);
	tr.tmp = TAKE(1024 * 8);
	double *ring = TAKE((size_t)CX_RING * CX_DB * CX_NBUS * 8);
	if (off > (8u << 20)) { fprintf(stderr, "train-gate: arena overflow %zu\n", off); return 1; }
	tr.ET = huge(4u << 20);
	tr.GET = huge(4u << 20);

	cx_col_init(tr.col.prm, 0);
	memset(tr.col.T, 0, CX_NT * 8);
	for (int i = 0; i < CX_DU; i++) tr.cg[i] = 1.0;
	double *E = malloc((size_t)CX_VOCAB * CX_DE * 8);
	cx_emb_init(E);
	for (int v = 0; v < CX_VOCAB; v++) for (int c = 0; c < CX_DE; c++) tr.ET[CX_ETB(v, c)] = E[(long)v * CX_DE + c];
	free(E);
	uint32_t dummy[CX_RING];
	cx_in_init(dummy, CX_RING, ring, 0);

	if (runm) return run_mode(argc, argv, &tr, ring);
	FILE *f = fopen(argv[2], "rb");
	if (!f) { perror(argv[2]); return 1; }
	uint16_t *tok = malloc((size_t)(warm + L + 1) * 2);
	if (fread(tok, 2, (size_t)(warm + L + 1), f) != (size_t)(warm + L + 1)) { fprintf(stderr, "train-gate: short token file\n"); return 1; }
	fclose(f);
	uint32_t *ids = malloc((size_t)(warm + L + 1) * 8);
	for (int t = 0; t < warm + L + 1; t++) ids[t] = tok[t];
#define BT(t) (ring + (long)((t) % CX_RING) * CX_DB * CX_NBUS)

	uint64_t c[12];
	c[0] = cx_rdtsc();
	cx_gather_prefetch(tr.ET, ids[0]);
	for (int t = 0; t < warm; t++) cx_train_fwd_step(&tr, 0, ids[t], ids[t + 1], BT(t));
	c[1] = cx_rdtsc();
	cx_train_chunk_begin(&tr);
	for (int t = 0; t < L; t++) cx_train_fwd_step(&tr, t, ids[warm + t], ids[warm + t + 1], BT(warm + t));
	c[2] = cx_rdtsc();
	cx_head_logits(&tr);
	c[3] = cx_rdtsc();
	cx_head_loss(&tr, ids + warm + 1);
	c[4] = cx_rdtsc();
	double mean = 0;
	for (int t = 0; t < L; t++) mean += tr.loss[t];
	printf("train-gate %s: cpu %d, L %d, warm %d; us per step: warm-up fwd %.2f, chunk fwd %.2f, head logits %.2f, softmax+loss %.2f; mean loss %.6f nats\n",
	       argv[1], sched_getcpu(), L, warm, US(c[0], c[1]) / (warm ? warm : 1), US(c[1], c[2]) / L, US(c[2], c[3]) / L, US(c[3], c[4]) / L, mean / L);
	if (dump(argv[3], "fwd-loss.f32", tr.loss, (size_t)L * 8) || dump(argv[3], "fwd-dl.f32", tr.logits, (size_t)L * CX_VOCAB * 8) ||
	    dump(argv[3], "fwd-hc.f32", tr.hc, (size_t)L * CX_DU * 8)) return 1;
	if (!bwd) return 0;

	cx_zero(tr.GET, (long)CX_VOCAB * CX_DE);
	c[5] = cx_rdtsc();
	cx_head_dhc(&tr);
	c[6] = cx_rdtsc();
	for (int t = L - 1; t >= 0; t--) cx_train_bwd_step(&tr, t, BT(warm + t));
	c[7] = cx_rdtsc();
	cx_train_param_grads(&tr);
	c[8] = cx_rdtsc();
	cx_head_dE(&tr);
	c[9] = cx_rdtsc();
	cx_train_input_grads(&tr, ids + warm);
	c[10] = cx_rdtsc();
	printf("train-gate bwd: us per step: head dhc %.2f, column bwd %.2f, param grads %.2f, head dE %.2f, input rows %.2f; total train step %.2f\n",
	       US(c[5], c[6]) / L, US(c[6], c[7]) / L, US(c[7], c[8]) / L, US(c[8], c[9]) / L, US(c[9], c[10]) / L,
	       (US(c[1], c[4]) + US(c[5], c[10])) / L);
	if (dump(argv[3], "gprm.f32", tr.gprm, CX_NPARAM * 8) || dump(argv[3], "gcg.f32", tr.gcg, CX_DU * 8) ||
	    dump(argv[3], "get.f32", tr.GET, (size_t)CX_VOCAB * CX_DE * 8) || dump(argv[3], "dhc.f32", tr.dhc, (size_t)L * CX_DU * 8)) return 1;
	return 0;
}
