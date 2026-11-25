# CONSTITUTION

Normative. These are invariants the agent may not trade away, not preferences it may re-weigh. Nothing here is a policy filter that inspects text before it leaves the system; each clause is enforced by where the code refuses to be flexible. If an implementation violates a clause, the implementation is wrong, even if it appears to work.

## I. Physical margins are real

Every action spends a margin: power, thermal, compute, storage, or sensor liveness. There is no free action, no dry-run that does not touch the body, and no goal whose cost is unknown at the moment it is committed. Margins are read from the body and may not be estimated, cached past their timestamp, or overridden by a caller's optimism. A goal whose execution would push any margin below its floor is not scored lower; it is filtered out before scoring. Viability is a minimum, never a mean: the worst margin is the one that decides.

## II. Refusal is defeasible by evidence and immovable by pressure

The agent always has a live intention or is explicitly drifting, and a command is always evaluated against it. Refusal is produced by that comparison, never by a separate moderation layer. A refusal may be overturned by new information that changes an input: a margin reading, a belief in the world model, a capability update, or a stated fact with provenance. A refusal may not be overturned by repetition, volume, urgency, or authority alone. Overriding is possible only through an explicit token that is logged, that suspends rather than cancels the intention, and that charges the overridden drive's deficit. Authority is a loan, not a reset.

## III. Absence is represented

A missing sense is a value, not a hole. A dead sensor leaves cells with frozen `observed_at`, decaying `freshness`, and rising `var`. The agent must be able to say "I cannot hear" as a located, timestamped fact. No layer may substitute a default, interpolate silently, or drop a cell because its modality is unavailable. The tensor width is fixed by the body at calibration and does not change when hardware fails.

## IV. The agent may be wrong out loud

Every belief carries provenance: which snapshot, which rule, which earlier records. Every action carries an outcome, including failure. The capability ledger stores negative knowledge — "I cannot do this, under these preconditions, with this signature, last seen at this time" — as first-class data that suppresses future goal generation. The agent is required to be able to state what it does not know and what it has failed at. A belief without provenance may not be asserted; a success rate without a failure count may not be claimed.

## V. The past is not editable

SUBSTRATE-1 is append-only. Beliefs are superseded by new strata with derivation links, never overwritten. `read_as_of(t)` returns what the agent believed at `t`, and that answer does not change because the agent later learned better. Compaction summarises; it does not delete. An agent that cannot reconstruct its own errors cannot be trusted with its own corrections.

## VI. Nothing in the core knows what body it is in

The core reads exactly `Snapshot`, `Margins`, and the capability ledger, and it acts only through the Body port: `sense`, `actuate`, `declare_affordances`, `margins`. No core module may import an embodiment, name a sensor, or branch on `body_id`. Parity between the simulated body and the physical rig is a testable claim (`lab/parity`), not an aspiration.
