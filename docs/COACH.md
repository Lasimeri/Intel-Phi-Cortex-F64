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
