# results (F64 port)

| file | what |
|---|---|
| s00c-isa-f64.md | the F64 ISA gate: every vpu.h function vs long double, 2^18 inputs per set, final build |
| s00c-isa-f64-wide.md | the same gate on the widened exp / exp2 sets before the overflow-edge fix (edge visible) |
| s01-col-f64.log | one F64 column, 2,000 steps on CPU 241: timing and the host comparison (col-ref) |
| s03c-bwd-f64.log | single-column training step on CPU 241 (train-gate bwd) and the gradient comparison (train-check) |
| s04c-cluster-f64.log | 120-column forward and backward gates and the ctrain-check comparisons (first NaN run, the arity failure and the final pass are all in the log in order) |
| s04c-diag-nan.log | the NaN diagnosis: 64-chunk run with per-chunk loss, the bus-off backward gate and its zero-bus host comparison, the single-column run |
| s04c-bisect-nan.log | 600-chunk runs with per-chunk loss: the first NaN at chunk 449 with the unguarded reciprocal, finite throughout with the guards |
| s04c-bisect-ckstat.log | ckstat of the finite checkpoint at chunk 450: parameter magnitudes per block |
| s04c-run-f64-vs-f32.log | the F32 cluster's 320k-token run on the same card with the arguments of the F64 run (the F64 part of this log is the NaN run before the guards) |
| s04c-run-f64.log | the F64 320k-token run with the final binary |
| s05c-gen-f64.log | the F64 generation daemon serving the 320k checkpoint |

The single-thread gates (s01, s03c) ran on CPU 241 while an F32 training run occupied cores 0 to 59; their timings
carry that contention. The cluster gates and the training runs ran on an otherwise idle card.
