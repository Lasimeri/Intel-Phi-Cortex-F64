# Intel Phi Cortex F64

The [Intel Phi Cortex](https://github.com/Lasimeri/Intel-Phi-Cortex) (120 small recurrent columns on one Xeon Phi 7120A,
two per core, voting into a shared head and trained on the card) ported from F32 to F64. Same shapes, same parameter
layout, same tokenizer, same mailbox and DMA protocol, same host references; every parameter, activation, gradient and
optimizer state is a double and every vector instruction is the 8-lane `pd` form.

Everything here is measured on the card. Numbers without a log path are not claimed.

## What F64 costs on Knights Corner

The KNC vector unit has F64 add, subtract, multiply, FMA, min/max, rounding and conversions, but its reciprocal, inverse
square root, exp2 and log2 instructions exist only for F32 (Intel 327364-001, `vrcp23ps`, `vrsqrt23ps`, `vexp223ps`,
`vlog2ps`). `card/vpu.h` builds the F64 forms from them:

| function | construction | max error vs long double (2^18 inputs) | log |
|---|---|---|---|
| `cx_rcp` | `vrcp23ps` on the narrowed input, 3 Newton steps | 1.24 ULP (median 0.27) | `results/s00c-isa-f64.md` |
| `cx_rsqrt` | `vrsqrt23ps`, 3 Newton steps | 1.52 ULP (median 0.29) | same |
| `cx_exp2` | n = round(x), degree-13 Taylor in (x - n) ln 2, exact 2^n from `vexp223ps` | 0.90 ULP (median 0.25), none over 1 ULP | same |
| `cx_exp` | log2 e split hi/lo, FMA argument reduction, then `cx_exp2` | 1.04 ULP (median 0.25), 6 of 262,144 over 1 ULP | same |
| `cx_log2` | `vlog2ps` estimate, cubic correction in w = x 2^-y0 - 1 | 3.7e-16 absolute on [0.5, 2]; near x = 1 the relative error is that of the absolute 1e-16 | same |
| `cx_sigmoid` | 1 / (1 + exp(-x)) | 2.88 ULP (median 0.41) over [-40, 40] | same |
| `cx_hsum`, `cx_hmax` | swizzles within 256-bit halves, `vpermf32x4` 0x4E across | exact (hmax), rounding of the sum order (hsum) | same |

A single F64 column alone (`col-gate`, 2,000 steps, one thread): 61.5 us min, 108.8 us mean per step, 866 to 889 L2
read misses per step, column 402,176 B (`results/s01-col-f64.log`). The F32 column: 22.7 us min, 26.0 us mean. The
F64 column no longer fits two per core in the 512 KiB L2 (2 x 402 KiB), which is where the slowdown beyond the 2x lane
count comes from.

The 120-column cluster (`cluster-gate`, L 32, bus on, idle card): forward 973 to 1,155 us per step, backward with
gradients 982 to 1,019 us per step, 9,248 to 9,261 L2 read misses per step per thread (`results/s04c-cluster-f64.log`).
The F32 cluster on the same card: forward 355 to 392 us, backward 395 to 403 us, 2,485 to 2,499 misses.

## Verification against the double host reference

| gate | what | result | log |
|---|---|---|---|
| wiring | 2,000 column steps, card vs `tools/col-ref.c` | u 1.78e-15, bc 2.78e-15, g 4.7e-16, T 9.8e-16 relative; step-0 intermediates 3e-16 to 2.5e-15 | `results/s01-col-f64.log` |
| training step | L 16 after 16 warm steps, card vs `tools/train-check.c` | loss 8.445217 both (5.3e-15 apart); dl 1.6e-17; gradients P_in 3.1e-15, Wqb 2.6e-15, gain 2.7e-15, W 2.4e-15, W1 2.3e-15, b1 2.3e-15, Wu 3.8e-15, cg 2.8e-15, E 1.8e-15, dhc 6.9e-15 relative | `results/s03c-bwd-f64.log` |
| 120-column forward | L 32 after 32 warm steps, bus on, card vs `tools/ctrain-check.c` (columns 0, 59, 119 dumped) | loss 8.594224 both (4.4e-15 apart); hc 9.1e-16; u, bc, g 2e-16 to 1.2e-15 relative | `results/s04c-cluster-f64.log` |
| 120-column backward | same, all parameter gradients of the dumped columns, cg and the embedding | 1.7e-15 to 5.0e-15 relative per block; cg 3.7e-15; E 3.0e-15 | same |
| 120-column backward, bus off | the same gate with CX_BUS=0 against the zero-bus reference (`CREF_BUS_ZERO`) | loss 8.577545 both; gradients 2.5e-15 to 4.0e-15; cg 2.4e-15; E 2.5e-15 | `results/s04c-diag-nan.log` |

The gates pass with gradients only. The first training runs then went NaN after a few hundred chunks while the F32
run on the same tokens stayed finite: the F64 reciprocal seeded from `vrcp23ps` returned NaN for an inf argument (the
Newton step computes 0 * (2 - inf * 0)), reached through the sigmoid once e^-x overflows, where F32 had returned the
seed 0. `cx_rcp` and `cx_rsqrt` now map inf to 0 and 0 to inf explicitly; the same 600-chunk run is finite
throughout (`results/s04c-bisect-nan.log`). The traps are collected in `docs/F64-NUMERICS.md`.

The same gates in the F32 port agree to about 1e-6. One finding from the F64 wiring gate applies to both ports and is
recorded in `lib/cx-col.h`: a scalar `float` product on KNC (no SSE) stays in extended precision, so storing it into a
`double` skips the F32 rounding the host performs. The shared initialisation is therefore computed exactly in double
(24-bit integer times a power of two).

## Training and generation next to the F32 port on the same card

320,000 TinyStories tokens, chunks of 32, Adam 0.9/0.95, lr 3e-3 with a 50-chunk warm-up, bus off, validation on
100,000 tokens (`results/s04c-run-f64.log`, F32 in `results/s04c-run-f64-vs-f32.log`):

| | F64 | F32 |
|---|---|---|
| training rate | 503 tok/s (fwd+vote+head 882 to 884 us, bwd+grads+adam 1,102 to 1,106 us per step) | 1,258 to 1,261 tok/s (323 to 324 us, 469 to 470 us) |
| validation loss after 320k tokens | 4.3539 nats = 6.2813 bits/token = 1.5858 bits/byte | 4.4617 nats = 6.4369 bits/token = 1.6251 bits/byte |
| wall | 636.1 s train + 88.0 s validation | 253.9 s + 31.7 s |
| checkpoint | 46,926,848 B | 23,463,424 B |

The F64 run ends 0.156 bits/token lower at the same token count. One run each; the spread across seeds is not
measured, so the difference is reported, not explained.

Generation (`cortexd` on the card, `host/cortex` on the host, `results/s05c-gen-f64.log`): checkpoint load by DMA
21.0 ms; greedy 976 to 981 tok/s overall (card 1,003 to 1,008 us per token, host 17 us); sampled 869 to 871 tok/s
(card 924 us, host sampler 232 us); different seeds give different text once the sampler keeps more than one candidate
(the default settings leave a near one-hot set on this model, so two seeds can coincide). The F32 daemon: 12.2 ms,
2,144 to 2,182 tok/s greedy, 1,383 to 1,489 sampled. The
320k-token model itself is early: greedy decoding loops after about 20 tokens (4-gram repetition 0.381 in the samples,
0.045 in human text), which is what the coach work in `docs/COACH.md` is for.

## Not yet measured

A seed spread of the F64 vs F32 validation difference; the F64 daemon with a longer-trained checkpoint; two instead of
three Newton steps in the reciprocal seeds.
## The coach

`docs/COACH.md`: a training run publishes chunk, loss, learning rate and greedy samples into the host-memory part;
`host/coach` decodes and scores the samples (`tools/coherence.h`, word and bigram lists built from the human corpus,
4-gram repetition inside 256-word windows) and writes back a learning-rate scale, a checkpoint request and a stop flag.
Gated on the card: sampling leaves every per-chunk loss identical; the knobs act from the next report. Nothing the coach
writes and nothing the model generates is ever training data.

## Layout

Identical to the F32 repository. `build.sh` builds the card programs with the Intel-Phi-LLVM S3 clang (`-march=knc`)
and the host tools with cc. Card binaries run from `/tmp/cortex64` on the card.
