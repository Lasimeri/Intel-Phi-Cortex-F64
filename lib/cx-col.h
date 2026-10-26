/*
 * cx-col.h (F64 port): one column's shapes, parameter layout, and the deterministic initialisation that the card (S3 clang,
 * -march=knc) and the host reference (gcc, double) both compile, so that both sides hold bit-identical F32 weights and
 * inputs without a file transfer. Plain C, no intrinsics. Every offset is a multiple of 16 doubles (two 64-byte lines).
 *
 * Column c, token t (PLAN.md "Column"):
 *   e   = E[id]                       E 4096 x 128 shared
 *   p   = e^T P_in                    P_in 128 x 64
 *   qb  = p^T Wqb                     Wqb 64 x 16
 *   s_j = qb . bus_j   (j < 120)      bus kept transposed, Bt[16][128]; slots 120..127 are zero and masked out
 *   alpha = softmax(s),  r = sum_j alpha_j bus_j                                       (16)
 *   x   = RMSNorm([p ; r]) * gain                                                      (80)
 *   y   = x^T W                       W 80 x 208: q 0..63, k 64..127, v 128..191, a_pre 192, b_pre 193, rest 0
 *   k   = k / |k|,  a = sigmoid(a_pre) (forget),  b = sigmoid(b_pre) (write strength)
 *   S_t = a S_{t-1} + b (v - a S_{t-1} k) k^T     kept as T = S^T, T[j][i] = S[i][j], so S k = sum_j k_j T[j]
 *   o   = S_t q = a (S_{t-1} q) + (k . q) d,  d = b (v - a S_{t-1} k)
 *   h   = [o ; x]^T W1 + b1           W1 144 x CX_DH (64; 96 before the Stage 1 shrink)
 *   z   = SiLU(h)
 *   u   = z^T Wu (vote, 128)   g = sigmoid(z . wg + bg) (confidence)   bc = z^T Wb (bus out, 16)
 */
#ifndef CX_COL_H
#define CX_COL_H

#include <stdint.h>

/* F64 port: every parameter, activation and state is a double; a 512-bit vector holds 8 lanes. The initial values are
 * the F32 port's values widened (same generator), so the two ports start from the same weights. */
typedef double cx_real;
#define CX_VL 8

#define CX_VOCAB 4096
#define CX_DE 128      /* embedding width */
#define CX_DP 64       /* p */
#define CX_DB 16       /* bus slot width */
#define CX_NBUS 128    /* bus slots held (120 columns + 8 zero slots) */
#define CX_NCOL 120
#define CX_DX 80       /* [p ; r] */
#define CX_DS 64       /* fast-weight dimension */
#define CX_NW 208      /* W columns, 194 used */
#define CX_DM 144      /* [o ; x] */
#ifndef CX_HID
#define CX_HID 64      /* MLP hidden: 64 since Stage 1 (96 did not leave two columns resident in L2; -DCX_HID=96 rebuilds the old shape) */
#endif
#define CX_DH CX_HID
#define CX_DU 128      /* vote */

/* parameter block, offsets in doubles */
#define CX_O_PIN 0                                   /* 128 x 64 */
#define CX_O_WQB (CX_O_PIN + CX_DE * CX_DP)          /* 64 x 16 */
#define CX_O_GAIN (CX_O_WQB + CX_DP * CX_DB)         /* 80 */
#define CX_O_W (CX_O_GAIN + CX_DX)                   /* 80 x 208 */
#define CX_O_W1 (CX_O_W + CX_DX * CX_NW)             /* 144 x 96 */
#define CX_O_B1 (CX_O_W1 + CX_DM * CX_DH)            /* 96 */
#define CX_O_WU (CX_O_B1 + CX_DH)                    /* 96 x 128 */
#define CX_O_WG (CX_O_WU + CX_DH * CX_DU)            /* 96 */
#define CX_O_BG (CX_O_WG + CX_DH)                    /* 16: bias in lane 0 */
#define CX_O_WB (CX_O_BG + 16)                       /* 96 x 16 */
#define CX_NPARAM (CX_O_WB + CX_DH * CX_DB)          /* 44,512 doubles at CX_DH 64 = 356,096 B */
#define CX_NT (CX_DS * CX_DS)                        /* fast weights, 4,096 doubles */

/* activations (doubles), one block per column */
#define CX_A_P 0
#define CX_A_QB 64
#define CX_A_S 80          /* 128 scores */
#define CX_A_ALPHA 208     /* 128 */
#define CX_A_MIN 336       /* [o ; x], 144 */
#define CX_A_Y 480         /* 208 */
#define CX_A_K 688         /* 64, unit-norm k */
#define CX_A_H 752         /* CX_DH */
#define CX_A_Z (CX_A_H + CX_DH)
#define CX_A_U (CX_A_Z + CX_DH)
#define CX_A_G (CX_A_U + CX_DU)   /* 16, all lanes g */
#define CX_A_BC (CX_A_G + 16)
/* saved for the backward (Stage 3): r, the norm scales, a, b, k.q, S k, S q, d, the consensus norm scale, hc, sigmoid(h) */
#define CX_A_R (CX_A_BC + 16)       /* 16 */
#define CX_A_XS (CX_A_R + 16)       /* 16, all lanes */
#define CX_A_KS (CX_A_XS + 16)      /* 16 */
#define CX_A_AV (CX_A_KS + 16)      /* 16, a in all lanes */
#define CX_A_BV (CX_A_AV + 16)      /* 16, b */
#define CX_A_KQ (CX_A_BV + 16)      /* 16 */
#define CX_A_SK (CX_A_KQ + 16)      /* 64 */
#define CX_A_SQ (CX_A_SK + 64)      /* 64 */
#define CX_A_D (CX_A_SQ + 64)       /* 64 */
#define CX_A_XSU (CX_A_D + 64)      /* 16 */
#define CX_A_HC (CX_A_XSU + 16)     /* 128 */
#define CX_A_SG (CX_A_HC + 128)     /* CX_DH */
#define CX_A_E (CX_A_SG + CX_DH)    /* 128: the input embedding row (training) */
#define CX_NACT (CX_A_E + CX_DE)

/* per-step gradient vectors saved by the backward for the chunk-end parameter gradients (doubles) */
#define CX_G_DU 0          /* 128 */
#define CX_G_DH 128        /* CX_DH (64) */
#define CX_G_DY 192        /* 208 */
#define CX_G_DQB 400       /* 16 */
#define CX_G_DP 416        /* 64 */
#define CX_G_DE 480        /* 128 */
#define CX_NG 608

/* what a gate step dumps: u, bc, g (160 doubles) */
#define CX_DUMP 160

#define CX_EPS 1e-6f
#define CX_LOG2E 1.4426950408889634f

/* bus ring for the gates: a step uses ring[t % CX_RING] */
#define CX_RING 64

/* splitmix64 (Steele, Lea, Flood 2014; the reference increment and mixers) */
static inline uint64_t cx_rng(uint64_t *s)
{
	uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
	return z ^ (z >> 31);
}

/* uniform in [-scale, scale): a 24-bit integer times scale / 2^23, both exact in double, so the product is exact and the
 * card and the host get the same bits whatever the scalar unit (KNC has no SSE: a float product may stay in extended
 * precision; computing in double removes the rounding question) */
static inline double cx_unif(uint64_t *s, double s23)
{
	return (double)((int32_t)(cx_rng(s) >> 40) - 8388608) * s23;
}

static inline void cx_fill(double *p, long n, uint64_t *s, float scale)
{
	const double s23 = (double)scale / 8388608.0;
	for (long i = 0; i < n; i++) p[i] = cx_unif(s, s23);
}

/* init scales: 1 / sqrt(fan in), as decimal literals so both compilers round them the same way */
#define CX_S_PIN 0.08838834764831845f   /* 1/sqrt(128) */
#define CX_S_WQB 0.125f                 /* 1/sqrt(64) */
#define CX_S_W 0.11180339887498948f     /* 1/sqrt(80) */
#define CX_S_W1 0.08333333333333333f    /* 1/sqrt(144) */
#if CX_DH == 96
#define CX_S_WU 0.10206207261596575f    /* 1/sqrt(96) */
#else
#define CX_S_WU 0.125f                  /* 1/sqrt(64) */
#endif
#define CX_S_E 0.1f

#define CX_SEED_E 0x45E3B0C7D1A2F9ull
#define CX_SEED_P(k) (0xC0A7E80000ull + 0x9E3779B97F4A7C15ull * (uint64_t)((k) + 1))
#define CX_SEED_IN(k) (0x1A2B3C4D0000ull + 0xD1B54A32D192ED03ull * (uint64_t)((k) + 1))

static inline void cx_emb_init(double *E)
{
	uint64_t s = CX_SEED_E;
	cx_fill(E, (long)CX_VOCAB * CX_DE, &s, CX_S_E);
}

static inline void cx_col_init(double *prm, int k)
{
	uint64_t s = CX_SEED_P(k);
	cx_fill(prm + CX_O_PIN, CX_DE * CX_DP, &s, CX_S_PIN);
	cx_fill(prm + CX_O_WQB, CX_DP * CX_DB, &s, CX_S_WQB);
	for (int i = 0; i < CX_DX; i++) prm[CX_O_GAIN + i] = 1.0f;
	for (int i = 0; i < CX_DX; i++)
		for (int j = 0; j < CX_NW; j++) prm[CX_O_W + i * CX_NW + j] = j < 194 ? cx_unif(&s, (double)CX_S_W / 8388608.0) : 0.0f;
	cx_fill(prm + CX_O_W1, CX_DM * CX_DH, &s, CX_S_W1);
	for (int i = 0; i < CX_DH; i++) prm[CX_O_B1 + i] = 0.0f;
	cx_fill(prm + CX_O_WU, CX_DH * CX_DU, &s, CX_S_WU);
	cx_fill(prm + CX_O_WG, CX_DH, &s, CX_S_WU);
	for (int i = 0; i < 16; i++) prm[CX_O_BG + i] = 0.0f;
	cx_fill(prm + CX_O_WB, CX_DH * CX_DB, &s, CX_S_WU);
}

/* gate inputs of column k: steps token ids, then CX_RING bus matrices Bt[16][128] with slots 120..127 zero */
static inline void cx_in_init(uint32_t *ids, int steps, double *ring, int k)
{
	uint64_t s = CX_SEED_IN(k);
	for (int t = 0; t < steps; t++) ids[t] = (uint32_t)(cx_rng(&s) >> 52);   /* 12 bits: 0..4095 */
	const double s23 = 1.0 / 8388608.0;
	for (int r = 0; r < CX_RING; r++)
		for (int d = 0; d < CX_DB; d++)
			for (int j = 0; j < CX_NBUS; j++) ring[((long)r * CX_DB + d) * CX_NBUS + j] = j < CX_NCOL ? cx_unif(&s, s23) : 0.0f;
}

#endif
