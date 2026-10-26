# The coach: guiding a training run from the host

Direction (2026-10-26): the model is trained under supervision that keeps it on course for coherent language, with
two hard rules. Nothing written by the supervisor is ever training data, and nothing the model generates is ever
training data. The coach controls the process (what the run trains on, in which order, at which learning rate, when it
stops, which checkpoint is kept) and measures the outcome; the gradient still comes only from the human corpus.

## Pieces

| piece | where | role |
|---|---|---|
| `tools/coherence.c` | host | builds a word list from the human training text and scores any text: word-hit share, 4-gram repetition, sentence length, word length. The reference scores of the human validation text set the target band. |
| data curation (rule based) | host, `tools/jsonl-text.c` and successors | English, ASCII, prose only, length bounds, duplicates removed; a curriculum order by a difficulty proxy. Rules, not generated text. |
| schedule | `cluster-gate run` | warm-up, decay, epochs, checkpoint every N chunks, validation-selected checkpoint. |
| live loop | mailbox in the host-memory part (`lib/cx-proto.h`), no control socket needed while the run holds it | the trainer publishes chunk, loss, lr and a greedy sample of fixed prompts every EVERY chunks; the host coach scores the sample, logs, and writes back a learning-rate scale, a save flag and a stop flag, read by the card once per chunk. |

## Control block (host-memory part, after the generation mailbox fields)

| offset | field | writer | meaning |
|---|---|---|---|
| CX_K_MAGIC | "COCH" | card | the trainer is live |
| CX_K_CHUNK | u64 | card | chunks done |
| CX_K_LOSS | f64 | card | mean training loss of the last EVERY chunks, nats per token |
| CX_K_LR | f64 | card | learning rate in effect |
| CX_K_SAMPLE | u16[2][64] | card | greedy continuation ids of the two fixed prompts, written every EVERY chunks |
| CX_K_SEQ | u64 | card | increments after each sample write |
| CX_K_SCALE | f64 | host | learning-rate multiplier applied from the next chunk on (default 1.0) |
| CX_K_FLAGS | u64 | host | bit 0 save a checkpoint now, bit 1 stop after the current chunk |

## Policy (rule based, the supervisor's judgement encoded once)

1. Keep the learning rate while the validation loss improves by at least 0.01 bits/token per evaluation window.
2. On a plateau, halve the learning rate; after two halvings without improvement, stop and keep the best checkpoint.
3. Treat a rising 4-gram repetition share or a falling word-hit share over three consecutive samples as a plateau
   signal even when the loss still falls (loss can fall by memorising frequent fragments).
4. Curriculum: short, high word-hit documents first, then the full set; the chat set after the story set.

## Honest ceiling

The model has 5,865,856 parameters and a 4,096-token byte-level BPE. Coherence is bounded by that capacity and by the
corpus. The coach can move the loss curve, avoid the plateaus and the memorisation failure modes, and select the best
checkpoint; it cannot add capacity. Every effect is measured against the uncoached run on the same card.

## Implemented and gated (2026-10-26 04:1x)

| piece | where | state |
|---|---|---|
| control block | `lib/cx-proto.h` CX_K_* (part + 4096..4504) | as specified above, plus CX_K_NSAMPLE and CX_K_ACK |
| trainer hooks | `tools/cluster-gate.c` run mode, `CX_COACH=N` | publishes chunk, loss, lr at each report; every N chunks a greedy 64-token continuation of two fixed prompts (`/tmp/cortex/prompt1.u16`, `prompt2.u16`, or CX_KPROMPT1/2) with the fast-weight state and the bus row saved and restored around it; reads scale and flags once per chunk behind one extra barrier; clears the magic at run end |
| host coach | `host/coach.c` | decodes samples with the BPE vocabulary, scores them with `tools/coherence.h` against a word and bigram list, logs, applies the policy; `--force-scale` and `--stop-at` are test knobs |
| metrics | `tools/coherence.c`, `tools/coherence.h` | `build`, `score`, `windows` (the human per-window band) |

Gates on the 7120A (`results/s06c-coach-gate.log`, `results/s06c-coach-b2.log`):

- A, no trajectory change: 200 chunks with sampling every 50 chunks against 200 chunks without: 0 of 200 per-chunk
  losses differ, validation identical (7.8758 bits/token). A sample costs about 0.13 s (two prompts, 64 tokens each).
- B, knobs: the coach goes live at chunk 0; the first report shows the unscaled rate 3.00e-3; a forced scale 0.5
  written after it shows as 1.50e-3 from the next report on; the stop flag set after the chunk-350 report ends the
  run at chunk 351 with the validation pass and the checkpoint still written; the coach exits when the card clears
  the block.
- A defect found by B and fixed: the card did not clear the magic at run end, so a coach started between runs read
  the previous run's block as live and acted at chunk 0. The card now clears magic and chunk; the coach counts a
  report only when the chunk number increases from a non-zero base.

Reference bands from the human text (`coherence windows`, 256-word windows): TinyStories validation (17,336 windows)
hit5 1.000, big 0.995 (p10 0.986), rep4 0.022 (p90 0.036); chat validation (224 windows) hit5 0.889 (p10 0.775),
big 0.594 (p10 0.434), rep4 0.027 (p90 0.055). The early model (100 to 300 chunks) loops on real words, which hit5
does not see (1.000) and rep4 does (0.90 to 0.97): both metrics are needed.

First coached run: `results/s06c-coach-run1.log` (host) and `results/s06c-coach-run1-card.log` (card), 320k tokens,
to be compared with the uncoached F64 run's validation loss of 6.2813 bits/token on the same tokens.

## Coached run 1 and the policy correction (04:2x)

320k-token budget, same tokens and seed as the uncoached F64 run, policy v1 (`results/s06c-coach-run1.log`,
`results/s06c-coach-run1-card.log`):

| event | chunk | cause |
|---|---|---|
| halving 1, scale 0.5 | about 1,500 | three greedy samples with rep4 above the human band 0.036 |
| halving 2, scale 0.25 | about 3,500 | the same rule |
| halving 3, scale 0.125 | about 4,500 | the same rule |
| stop | 5,001 | window gain -0.04 bits/token after the three halvings |

Validation at the stop, 160k tokens: 4.2946 nats = 6.1959 bits/token, against 6.2813 for the uncoached run at 320k.
The per-report losses of the two logs are not comparable (500-chunk means against 100-chunk means); the validation
pass on the same 100,000 tokens is.

What the log shows about the rules: greedy decoding of this model loops at every stage (rep4 0.74 to 0.97 from the
first sample to the last), so an absolute threshold taken from human text fires on every sample and decays the rate
for the wrong reason; the stop then followed the coach's own decay. The gain over the uncoached run is consistent with
learning-rate decay as such, not with the coach having detected anything. Policy v2 therefore takes rate decisions
from the loss windows only, uses the samples as a trend signal (a sample whose rep4 exceeds the mean of the previous
three by more than 0.1, once per run), and stops only at the floor after two consecutive windows with a rising loss.
The control for the number above is an uncoached run stopped at the same 160k tokens
(`results/s06c-control-160k.log`); the coached v2 run over the full 320k budget follows.
