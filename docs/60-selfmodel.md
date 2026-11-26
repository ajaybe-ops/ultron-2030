# 60 — Self-model

## Status

Implemented and tested: `cortex/selfmodel/` records every attempt in
SUBSTRATE-1, answers `CAPABILITY_PRESENT` / `CAPABILITY_ABSENT` with a
recency-weighted confidence, and survives a reboot through `boot_selfmodel()`.
Proof: [`tests/test_selfmodel.py`](../tests/test_selfmodel.py), with
cross-layer parity in [`tests/test_layer_parity.py`](../tests/test_layer_parity.py).

Not yet implemented: introspection beyond capability (the agent cannot yet
answer "what am I like" or "why did I do that"), a narrative self-report, and
the dialogue layer that would speak either. The capability ledger remains the
arbiter of *affordances*; the self-model is the answer to *can I?* for the
refusal machine.

## What it is

The self-model is an append-only memory of attempts, keyed by verb. Each
attempt carries the verb, whether it succeeded, the body's failure signature,
and the tick. The model is not a score and not a belief about the world; it is
a belief about the agent's own procedure set.

The default is absence. A verb with no successful attempt is
`CAPABILITY_ABSENT`, even if it has never been tried. A procedure that has
never worked is not a capability.

## Confidence

A present capability has a confidence in `[0, 1]`:

```
base      = successes / (successes + failures + 1)
recency   = 0.5 ** (max(0, now - last_success_tick) / 100)
confidence = round(base * recency, 6)
```

Confidence rises with repetition because `base` approaches 1, and decays with
recency because a success from a hundred ticks ago counts half as much as one
from now. It never reaches 1: there is always one more possible failure.

## Persistence

Every attempt is appended to SUBSTRATE-1 with `kind="selfmodel"` and payload
`{verb, ok, signature, tick}`. `SelfModel.from_substrate()` rebuilds the model
and attaches it for further appends; `SelfModel.from_records()` rebuilds
without a substrate, which is what replay uses. Replay must never write.

The session loop passes the model into `agent_tick()`; the tick payload carries
the model snapshot, so the same replay harness that checks intentions and drive
states also checks what the agent believed about itself.

## Query contract

`SelfModel.query(verb, tick)` is pure: it reads attempts and computes a view,
and does not append, cache, or mutate. `SelfModel.snapshot(tick)` returns a
detached `SelfModelSnapshot`, which is the only self-model object the refusal
machine receives. A decision therefore cannot change the model as a side
effect of being evaluated.
