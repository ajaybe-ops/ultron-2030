# 50 — SUBSTRATE-1 on-disk format

Normative. Version string: `SUBSTRATE-1`. A reader that does not find exactly this string in `<root>/FORMAT` must refuse to open the store. There is no migration path in this version.

## 1. Layout

```
<root>/FORMAT                          # b"SUBSTRATE-1\n"
<root>/shards/shard-000000.jsonl       # append shard
<root>/shards/shard-000001.jsonl
<root>/shards/compact-000000.jsonl     # compaction stratum
<root>/shards/shard-000000.superseded.json
```

Shards are newline-delimited JSON: one record per line, `\n` terminated, UTF-8, keys sorted, no whitespace between tokens. Append shards are named `shard-NNNNNN.jsonl` with six-digit zero padding. Compaction strata are named `compact-NNNNNN.jsonl` and are numbered independently of append shards.

## 2. Record schema

Exactly six fields, all required, no extras:

| Field | Type | Semantics |
|---|---|---|
| `seq` | int ≥ 1 | Global, gap-free, assigned by the store. Strictly increasing across every shard. |
| `t_mono` | float | Monotonic seconds at write time. The ordering clock. |
| `t_wall` | float | Wall-clock seconds. Only for human-facing statements. |
| `kind` | string | Non-empty. Namespaces the payload (`tick`, `session`, `belief`, …). |
| `provenance` | object | Why the record exists. Always contains `shard` (the file it lives in); carries `rule`, `source`, `derived_from`, `snapshot_digest` as applicable. |
| `payload` | object | Writer-defined content. Opaque to the store. |

`seq` is assigned by `append()`; a caller may pass the exact next value and nothing else. The store rejects gaps because a gap makes "everything up to here" ambiguous.

## 3. As-of reads

`read_as_of(t)` returns every record with effective time ≤ `t`, in `seq` order, where effective time is folded:

```
eff(0) = t_mono(0)
eff(i) = max(eff(i-1), t_mono(i))
```

A backwards clock jump therefore collapses into the previous instant instead of reordering history or dropping records. Records written after `t` are excluded; records superseded by a later compaction are still included, because at time `t` they were the truth. `skew()` reports `max_backward_jump` and cumulative `wall_mono_drift` so the damage is visible rather than silent.

`read_as_of` never means "latest". Code that wants the current view calls `current()`, which excludes superseded records.

## 4. Compaction

`compact(kind)` selects every record of `kind` that is not already superseded, keeps the last record per `payload["key"]` (falling back to `seq:<n>` when no key exists), and writes the survivors to a new `compact-NNNNNN.jsonl` stratum.

Each survivor:

- keeps its original `payload`,
- gains `provenance.rule = "compaction"`,
- gains `provenance.stratum = <n>`,
- gains `provenance.derived_from = [seq, ...]`, listing **every** original record it stands for, in ascending order,
- gains `provenance.shard = <new stratum file name>`,
- receives a fresh `seq` and a `t_mono` strictly greater than every existing record.

Compaction is append-only with respect to truth: it never edits a file in place and never deletes one. It writes a marker per source shard, `<shard>.superseded.json`:

```json
{"format":"SUBSTRATE-1","shard":"shard-000000.jsonl",
 "superseded_by":"compact-000000.jsonl","superseded_seqs":[1,2,3],
 "shard_fully_superseded":false,"stratum":1,"at_seq":9,"first_seq":8}
```

Markers are merged if compaction runs twice over the same shard. `current()` drops records named in `superseded_seqs`, and skips a shard entirely when `shard_fully_superseded` is true, so a shard holding several kinds is not lost when only one kind is compacted. Compacting an already-compacted kind produces a second stratum whose `derived_from` points at the first stratum's seqs: that is the intended history of belief revision.

## 5. Crash safety

Appends are a single `write()` of `<json>\n` followed by `fsync`. A crash can therefore leave at most one incomplete line, at the end of the highest-numbered shard.

- A final line without a terminating newline, or one that fails to parse, is ignored on read.
- On open, that torn tail is truncated to the last newline so the next append does not concatenate onto a fragment. No parseable record is removed.
- A torn line in any shard that is not the final one raises `CorruptionError`. Only the tail can be interrupted; damage elsewhere is not a crash artefact and must not be silently tolerated.

## 6. Invariants a conforming reader may rely on

1. `seq` is unique and gap-free across all shards, including compaction strata.
2. Every record carries `provenance.shard`, and that file exists.
3. A `derived_from` list is non-empty and names only `seq` values that exist.
4. `read_as_of(t)` is stable: later writes never change its answer for the same `t`.
5. No compaction step reduces the number of files under `shards/`.
