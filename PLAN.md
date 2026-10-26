# Intel Phi Cortex F64: plan and status

The F64 port of the Intel Phi Cortex (https://github.com/Lasimeri/Intel-Phi-Cortex). Same architecture, shapes,
parameter layout, tokenizer, host references, mailbox and DMA protocol; every value a double, every vector instruction
the 8-lane `pd` form. This file records what was changed, the gates each change had to pass on the 7120A, and what is
measured. Design notes for the model itself are in the F32 repository's PLAN.md.

## Port rules

1. One source of shapes: `lib/cx-col.h` keeps the F32 element layout; `cx_real` is `double`, `CX_VL` is 8.
2. Every kernel is re-derived for 8 lanes, not search-and-replaced: vector counts (16 doubles = 2 vectors), masked
   stores (`__mmask8`, `(i & 7)`, `(i & ~7)`), per-step strides of 8-lane save slots, and the call-site arities of
   `cx_gemv*`, `cx_gemvt*`, `cx_outer*`, `cx_norm_bwd`. The single bug that survived to the cluster gate was a call-site
   arity (`cx_norm_bwd(..., 8, ...)` left from F32) plus a `t * 16` stride: all per-column gradients off by 300 to 2,400x
   while every forward quantity matched.
3. Transcendentals: KNC has none in F64. `card/vpu.h` builds them (docs/F64-NUMERICS.md) and `isa` + `isa-host` measure
   them against long double before anything else runs.
4. Shared initial values are computed exactly in double (a scalar `float` product on KNC stays in extended precision and
   a store into a double skips the rounding the host does).
5. Byte sizes: every `* 4` became `* 8`; arenas 4 -> 8 MiB per thread, embedding blocks 2 -> 4 MiB, the staged checkpoint
   23,463,424 -> 46,926,848 B, the DMA load 3 -> 5 pieces, logits 16 -> 32 KiB.

## Gates (all on the 7120A, raw logs in results/)

| gate | command | pass criterion | result |
|---|---|---|---|
| ISA | `isa-host gen`, `isa OP`, `isa-host check` | each F64 function near 1 ULP of long double over its set | rcp 1.24, rsqrt 1.52, exp2 0.90, exp 1.04, sigmoid 2.88 ULP; log2 3.7e-16 abs on [0.5, 2]; hmax exact (`results/s00c-isa-f64.md`) |
| wiring | `col-gate OUT 2000 1 0`, `col-ref` | card column == host double column | u 1.78e-15, bc 2.78e-15, T 9.8e-16 (`results/s01-col-f64.log`) |
| training step | `train-gate bwd valid.u16 OUT 16 16`, `train-check` | all gradients == host double | 1.8e-15 to 6.9e-15 relative (`results/s03c-bwd-f64.log`) |
| cluster | `cluster-gate fwd|bwd valid.u16 OUT 32 32 0 59 119`, `ctrain-check`, bus on and off | 120-column forward and backward == host double cluster | loss identical (4.4e-15), hc 9.1e-16, gradients 1.7e-15 to 5.0e-15, cg 3.7e-15, E 3.0e-15 (`results/s04c-cluster-f64.log`); bus off 2.5e-15 to 4.0e-15 (`results/s04c-diag-nan.log`) |
| training run | `CX_BUS=0 cluster-gate run train10m.u16 run 32 320000 3e-3 valid.u16 100000 500`, then the F32 binary with the same arguments | rate and validation loss next to the F32 baseline on the same card; the first two F64 runs went NaN (`docs/F64-NUMERICS.md`, traps five and six) | F64 6.2813 bits/token at 503 tok/s (`results/s04c-run-f64.log`); F32 6.4369 at 1,258 to 1,261 tok/s (`results/s04c-run-f64-vs-f32.log`) |
| generation | `cortexd` on the card, `cortex CKPT "prompt" N --bus 0` on the host | the daemon serves the F64 checkpoint over the mailbox and DMA | checkpoint load 21.0 ms, greedy 976 to 981 tok/s, sampled 869 to 871 tok/s; seeds verified to vary the text (`results/s05c-gen-f64.log`) |

## Measured cost of F64 on this card

| measurement | F64 | F32 |
|---|---|---|
| one column alone, per step (min / mean) | 61.5 / 108.8 us | 22.7 / 26.0 us |
| 120 columns forward, per step | 973 to 1,155 us | 355 to 392 us |
| 120 columns backward + gradients, per step | 982 to 1,019 us | 395 to 403 us |
| L2 read misses per step per thread | 9,248 to 9,261 | 2,485 to 2,499 |

A column is 402,176 B in F64; two per core exceed the 512 KiB L2.

## Open

- Two Newton steps instead of three in `cx_rcp` and `cx_rsqrt` (bit count says 46 -> 92 bits suffices): unmeasured.
- A log1p path for `cx_log2` near 1 if relative accuracy there is ever needed (the model only takes log2 of sums >= 1).
- The F64 column does not fit two per core in L2; a hidden size of 32 or one column per core would be the F64-native
  configuration. Not explored.
