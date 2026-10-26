/*
 * cx-proto.h: Cortex's part of the shared host-memory region. The host maps it from /dev/shm/phi-hostmem, the card
 * from /dev/phihost (uncached). Little-endian; offsets in bytes from the part's start. Host-written and card-written
 * fields sit on separate 64-byte lines; each side only polls lines the other side writes.
 *
 * The part sits at 1344 MiB of the 2 GiB region, 128 MiB long (ends at 1472 MiB). Neighbours (from their protocol
 * headers, 2026-10-25): below ends near 1304 MiB, above starts at 1536 MiB.
 */
#ifndef CX_PROTO_H
#define CX_PROTO_H

#include <stdint.h>

#define CX_MAGIC 0x58545243u       /* "CRTX" */
#define CX_VERSION 1
#define CX_PART_MIB 1344
#define CX_PART_SIZE (128u << 20)

/* written by the host */
#define CX_H_MAGIC 0               /* u32 */
#define CX_H_VERSION 4             /* u32 */
#define CX_H_SESSION 8             /* u64 raised by each host open */
#define CX_H_CMD 64                /* u64 step command: sequence in bits 0-47, argument in bits 48-63 */

/* written by the card */
#define CX_C_SESSION 128           /* u64 the session the card has taken */
#define CX_C_STATE 136             /* u32 0 starting, 1 ready, 2 error, 3 stopped */
#define CX_C_ACK 192               /* u64 the last command sequence finished */
#define CX_C_HEART 256             /* u64 counted up while the card side runs */

#define CX_CMD_SEQ(c) ((c) & 0xFFFFFFFFFFFFull)
#define CX_CMD_ARG(c) ((uint32_t)((c) >> 48))
#define CX_CMD(seq, arg) ((uint64_t)(seq) | (uint64_t)(arg) << 48)
#define CX_CMD_QUIT 0xFFFFFFFFFFFFull   /* sequence value that stops a gate program */

#define CX_DATA (64u << 10)        /* card -> host payload area (Stage 0e gate) */

/* Stage 5: the generation daemon (card/cortexd.c, host/cortex.c) */
#define CX_OP_LOAD 1               /* arg: unused; the checkpoint (120 x CX_NPARAM + 128 + 4096 x 128 floats) is staged at CX_STAGE */
#define CX_OP_RESET 2              /* every column's fast weights and the bus to zero */
#define CX_OP_STEP 3               /* arg: the token id; the card answers with the 4096 logits at CX_LOGITS, then the ACK */
#define CX_OP_BUS 4                /* arg: 1 bus on, 0 off (must match the checkpoint's training) */
#define CX_OP_EXIT 5               /* the daemon ends this session and exits (frees the control socket) */
#define CX_H_CMD_OP 72             /* u32 the op code of the current command (written before CX_H_CMD) */
#define CX_LOGITS (128u << 10)     /* 4096 F64 (32 KiB), written by the card's DMA; CX_C_ACK is the DMA status word */
#define CX_STAGE (1u << 20)        /* the staged checkpoint (F64: 46,926,848 B), host-written before CX_OP_LOAD */

#endif

/* ---- the coach's control block (docs/COACH.md): a training run publishes its state and reads the host's knobs here.
 * Offsets are bytes from the part start, below CX_DATA (64 KiB) and clear of the generation mailbox fields above. */
#define CX_K_MAGIC 4096           /* u32 "COCH" while a run is live, cleared on exit */
#define CX_K_CHUNK 4104           /* u64 chunks done (card) */
#define CX_K_LOSS 4112            /* f64 mean training loss of the last report window, nats/token (card) */
#define CX_K_LR 4120              /* f64 learning rate in effect (card) */
#define CX_K_SEQ 4128             /* u64 increments after each sample write (card) */
#define CX_K_NSAMPLE 4136         /* u32 tokens per sample row (card) */
#define CX_K_SAMPLE 4160          /* u16[2][64]: greedy continuations of the two fixed prompts (card) */
#define CX_K_SCALE 4480           /* f64 learning-rate multiplier applied from the next chunk on, default 1.0 (host) */
#define CX_K_FLAGS 4488           /* u64 bit 0 save a checkpoint now, bit 1 stop after the current chunk (host) */
#define CX_K_ACK 4496             /* u64 the host's flags word as last acted on (card) */
#define CX_K_MAGIC_VAL 0x48434f43u /* "COCH" little-endian */
