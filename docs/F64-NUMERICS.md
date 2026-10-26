# F64 transcendentals on Knights Corner

Knights Corner (Xeon Phi 7120A) has F64 add, subtract, multiply, FMA, min, max, rounding and conversions as 8-lane
`pd` instructions, but its estimate instructions exist only in F32: `vrcp23ps`, `vrsqrt23ps`, `vexp223ps`, `vlog2ps`
(Intel 327364-001). `card/vpu.h` builds the F64 functions from them. Every number below is measured on the card:
`build/host/isa-host gen DIR`, `isa OP DIR/OP.in DIR/OP.out` for each op on the card, `isa-host check DIR`
(2^18 doubles per set, judged against glibc long double). Raw output: `results/s00c-isa-f64.md`.

## Construction

Seed with the F32 instruction on the narrowed argument (`vcvtpd2ps`), widen (`vcvtps2pd`), finish in F64 FMA arithmetic.

| function | construction | max error | median | notes |
|---|---|---|---|---|
| `cx_rcp` | `vrcp23ps` seed, 3 Newton steps r = r (2 - x r) | 1.24 ULP | 0.27 | 23 -> 46 -> 92 bits: two steps should suffice; not measured |
| `cx_rsqrt` | `vrsqrt23ps` seed, 3 Newton steps r = r (1.5 - x r^2 / 2) | 1.52 ULP | 0.29 | same |
| `cx_exp2` | n = round(x) (`vrndfxpntpd`), f = x - n, 2^f = e^(f ln 2) by a degree-13 Taylor series in Horner form (u^14 / 14! < 5e-17 at the largest |u| = 0.347), times 2^n | 0.90 ULP, none over 1 ULP | 0.25 | degree 11 gave 55 ULP |
| `cx_exp` | log2 e = Lhi + Llo (1.4426950408889634 + 2.0355273740931033e-17); n = round(x Lhi); f = fma(x, Lhi, -n) + x Llo; then as `cx_exp2` | 1.04 ULP, 6 of 262,144 over 1 ULP on [-40, 40] | 0.25 | a single rounding of x log2 e costs about |x| 2^-53 relative: 72 ULP at |x| = 40. The hi/lo split alone still left 22.8 ULP; the FMA residual, which rounds once on a small quantity, is the part that works |
| `cx_log2` | y0 = `vlog2ps` seed; w = x 2^-y0 - 1 (|w| < 3e-5); y = y0 + (w - w^2/2 + w^3/3) / ln 2 | 3.68e-16 absolute on [0.5, 2] | 0.08 ULP | the linear step alone left a second-order error (median 14 ULP, p99.9 2,489 ULP). Near x = 1 the relative error is large because log2 x -> 0 while the absolute error stays about 1e-16; relative accuracy there needs a log1p path |
| `cx_sigmoid` | `cx_rcp`(1 + `cx_exp`(-x)) | 2.88 ULP on [-40, 40] | 0.41 | |
| `cx_hsum`, `cx_hmax` | swizzles cdab and badc act within 256-bit halves; `vpermf32x4` 0x4E swaps the halves; three steps | exact (hmax); summation order (hsum) | | all 8 lanes carry the result |

## Two traps, found by the 120-column gate and not by the single-column gates

The first F64 cluster forward produced a NaN loss (`results/s04c-cluster-f64.log`, first run).

1. **Huge arguments break the FMA reduction.** The cluster pads unused softmax slots with -1e300. For |x| ~ 1e300,
   n = round(x Lhi) is the rounded product itself and fma(x, Lhi, -n) is the product's rounding error, up to
   0.5 ulp(1e300) ~ 1e284. The polynomial overflows to inf, and inf times 2^n = 0 gives NaN. The F32 code padded with
   -3e38 and survived because F32 overflow to -inf saturated cleanly in `vcvtfxpntps2dq`. Fix: clamp the argument into
   the exp range before the reduction (`vgmaxpd`, `vgminpd`) and zero the lanes below the underflow threshold with a
   `vcmppd` mask.
2. **2^n taken from `vexp223ps` has the F32 exponent range.** Its Q8.24 input saturates at [-128, 128): n < -126 gives
   0 (F32 denormals flush to zero), n > 127 gives inf. exp(x) built on it is therefore wrong for x > 88.7 (inf instead
   of 2.7e43 at x = 100) and for x < -87.3 (0). Inside softmax and sigmoid this is harmless (terms below 1e-38 against
   sums of at least 1), but it is not an F64 exp. Fix: build 2^n in the double exponent field with 32-bit integer
   instructions (KNC has no 64-bit integer add or shift): `vcvtfxpntpd2dq` (8 int32 in the low 256 bits), `vpermd` to the
   odd 32-bit lanes, `vpaddd` 1023, `vpslld` 20, zero the even lanes (masked `vmovdqa32`, mask 0x5555), reinterpret as pd.

## A third trap, from the wiring gate

A scalar `float` product on KNC (no SSE; this toolchain keeps scalar float in extended precision) stored into a `double`
skips the F32 rounding the host performs. The shared initialisation generator did exactly that and the card's weights
differed from the host's at 2^-24 (the gate agreed only to 5.7e-8). `lib/cx-col.h` now computes the initial values
exactly in double (24-bit integer times a power of two). Values that must be bit-identical between card and host are
generated with exact arithmetic or rounded through a `volatile float`, never through an implicit float expression.

## Cost of F64 on this card

| measurement | F64 | F32 | log |
|---|---|---|---|
| one column alone, per step (min / mean) | 61.5 / 108.8 us | 22.7 / 26.0 us | `results/s01-col-f64.log`, F32 `results/s01-col.md` |
| 120 columns forward, per step | 990.9 (fwd gate), 1,155.1 (bwd gate) us | 355 to 392 us | `results/s04c-cluster-f64.log` |
| 120 columns backward + gradients, per step | 986.7 us | 395 to 403 us | same |
| L2 read misses per step per thread | 9,238 to 9,253 | 2,485 to 2,499 | same |

A column is 402,176 B in F64; two per core exceed the 512 KiB L2, which is where the slowdown beyond the 2x lane count
comes from.

## A fourth trap, from the training run

The seeds of `cx_rcp` and `cx_rsqrt` are taken on the argument narrowed to F32, so they hold only for arguments in the
F32 normal range (1.18e-38 to 3.4e38). Below it the seed is inf and the Newton steps turn it into NaN. The F64 Adam
first used s = v + 1e-300 for the inverse square root; every parameter whose gradient is exactly zero in a chunk (the
embedding rows of tokens not yet seen, the stop-gradient bus matrix) has v = 0, got s = 1e-300, and became NaN in the
first update, which made the training loss NaN from the second chunk on while every gate (gradients only, no Adam)
passed. The constant is 1e-30 now: it keeps s in range and changes the denominator sqrt(v) + eps by at most 5e-16 when
sqrt(v) >= 1e-15, far below eps = 1e-8. The range contract is stated in `card/vpu.h`; a full-range F64 reciprocal would
need exponent scaling of the argument before the seed (an extraction of the exponent field with the same 32-bit
integer route that `cx_pow2i` uses), which no caller in the model needs.

## A fifth trap, from the long training run: inf into the Newton step

The 320k-token F64 run went NaN at chunk 449 while the F32 run on the same tokens and schedule stayed finite. The seed
`vrcp23ps` of +inf is 0, as in F32, but the F64 Newton step then computes 0 * (2 - inf * 0) = NaN, where F32 simply
returned the seed. The input reaches inf through the sigmoid, 1 / (1 + e^-x), once a pre-activation falls below about
-709 and e^-x overflows: F32 gave sigmoid = 0 there, F64 gave NaN, and one NaN in a gate value spreads through the
consensus vector to every column within the chunk. `cx_rcp` and `cx_rsqrt` now map +-inf to 0 and 0 to +inf explicitly
(two compares and blends each). With the guards the same 600-chunk run is finite throughout (`results/s04c-bisect-nan.log`,
second run), validation 7.3543 bits/token after 19,200 tokens.

A sixth, not numerical: every checkpoint `fwrite`/`fread` in the gates still had element size 4 (the port's size fixes
matched `* 4` but not `, 4,`), so the F64 checkpoints written before 03:45 held the first half of each double array.
