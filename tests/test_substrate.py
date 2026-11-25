"""SUBSTRATE-1: append, as-of reads, compaction, crash safety, clock skew."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from substrate.format.substrate import (
    CorruptionError,
    FormatError,
    RecordError,
    Substrate,
)


def _record(t_mono: float, kind: str = "note", t_wall: float | None = None, **payload) -> dict:
    return {
        "t_mono": t_mono,
        "t_wall": 1_000.0 + t_mono if t_wall is None else t_wall,
        "kind": kind,
        "provenance": {"rule": "test", "source": "pytest"},
        "payload": payload or {"key": f"k{t_mono}"},
    }


def test_append_read_round_trip(sub_path: Path) -> None:
    store = Substrate(sub_path)
    written = [store.append(_record(t)) for t in (1.0, 2.0, 3.0)]

    assert [r.seq for r in written] == [1, 2, 3]
    read = store.read_all()
    assert [r.seq for r in read] == [1, 2, 3]
    assert [r.t_mono for r in read] == [1.0, 2.0, 3.0]
    assert read[0].provenance["rule"] == "test"
    assert read[0].payload["key"] == "k1.0"
    assert (sub_path / "FORMAT").read_text(encoding="utf-8").strip() == "SUBSTRATE-1"


def test_as_of_reads_historical_state(sub_path: Path) -> None:
    store = Substrate(sub_path)
    for t in (1.0, 2.0, 3.0):
        store.append(_record(t))

    assert [r.t_mono for r in store.read_as_of(0.5)] == []
    assert [r.t_mono for r in store.read_as_of(2.0)] == [1.0, 2.0]
    assert [r.t_mono for r in store.read_as_of(99.0)] == [1.0, 2.0, 3.0]


def test_as_of_is_stable_after_later_appends(sub_path: Path) -> None:
    store = Substrate(sub_path)
    for t in (1.0, 2.0):
        store.append(_record(t))
    before = [r.seq for r in store.read_as_of(2.0)]

    store.append(_record(9.0))

    assert [r.seq for r in store.read_as_of(2.0)] == before
    assert len(store.read_all()) == 3


def test_compaction_preserves_derivation_links(sub_path: Path) -> None:
    store = Substrate(sub_path)
    originals = [
        store.append(_record(t, kind="belief", key="light", value=v))
        for t, v in ((1.0, 0.2), (2.0, 0.4), (3.0, 0.6))
    ]
    store.append(_record(3.5, kind="belief", key="sound", value=0.9))

    written = store.compact("belief")

    by_key = {r.payload["key"]: r for r in written}
    assert set(by_key) == {"light", "sound"}
    assert by_key["light"].payload["value"] == 0.6
    assert by_key["light"].provenance["derived_from"] == [o.seq for o in originals]
    assert by_key["light"].provenance["rule"] == "compaction"
    assert by_key["sound"].provenance["derived_from"] == [4]


def test_compaction_never_deletes_and_marks_superseded(sub_path: Path) -> None:
    store = Substrate(sub_path)
    store.append(_record(1.0, kind="belief", key="light", value=0.2))
    store.append(_record(2.0, kind="belief", key="light", value=0.5))
    shards_before = set(p.name for p in (sub_path / "shards").glob("*.jsonl"))

    store.compact("belief")

    shards_after = set(p.name for p in (sub_path / "shards").glob("*.jsonl"))
    assert shards_before <= shards_after
    assert len(shards_after) == len(shards_before) + 1
    markers = list((sub_path / "shards").glob("*.superseded.json"))
    assert markers, "compaction must leave a supersession marker"
    marker = json.loads(markers[0].read_text(encoding="utf-8"))
    assert marker["superseded_by"] in shards_after
    assert marker["superseded_seqs"] == [1, 2]
    # history is still readable exactly as it was
    assert [r.seq for r in store.read_as_of(2.0)] == [1, 2]


def test_compaction_supersedes_current_view(sub_path: Path) -> None:
    store = Substrate(sub_path)
    store.append(_record(1.0, kind="belief", key="light", value=0.2))
    store.append(_record(2.0, kind="belief", key="light", value=0.7))
    store.append(_record(2.5, kind="tick", key="tick-1"))

    store.compact("belief")

    current = store.current()
    beliefs = [r for r in current if r.kind == "belief"]
    ticks = [r for r in current if r.kind == "tick"]
    assert len(beliefs) == 1
    assert beliefs[0].payload["value"] == 0.7
    assert len(ticks) == 1, "compaction must not swallow unrelated kinds"


def test_second_compaction_builds_a_new_stratum(sub_path: Path) -> None:
    store = Substrate(sub_path)
    store.append(_record(1.0, kind="belief", key="light", value=0.2))
    first = store.compact("belief")
    second = store.compact("belief")

    assert len(first) == 1
    assert len(second) == 1
    assert second[0].provenance["derived_from"] == [first[0].seq]
    assert len(store.current()) == 1


def test_truncated_final_line_is_ignored(sub_path: Path) -> None:
    store = Substrate(sub_path)
    store.append(_record(1.0))
    store.append(_record(2.0))
    shard = sorted((sub_path / "shards").glob("shard-*.jsonl"))[-1]
    with shard.open("ab") as handle:
        handle.write(b'{"seq":3,"t_mono":3.0,"kind":"no')

    reopened = Substrate(sub_path)
    assert [r.seq for r in reopened.read_all()] == [1, 2]
    assert reopened.append(_record(3.0)).seq == 3


def test_torn_line_outside_the_tail_raises(sub_path: Path) -> None:
    store = Substrate(sub_path, shard_size=1)
    store.append(_record(1.0))
    store.append(_record(2.0))
    first = sorted((sub_path / "shards").glob("shard-*.jsonl"))[0]
    with first.open("ab") as handle:
        handle.write(b'{"seq":99')

    with pytest.raises(CorruptionError):
        Substrate(sub_path, shard_size=1)


def test_backward_clock_folds_instead_of_dropping(sub_path: Path) -> None:
    store = Substrate(sub_path)
    for t in (1.0, 2.0, 1.5, 3.0):
        store.append(_record(t))

    # the 1.5 reading folds up to 2.0 rather than rewinding history
    assert [r.t_mono for r in store.read_as_of(2.0)] == [1.0, 2.0, 1.5]
    assert [r.t_mono for r in store.read_as_of(1.9)] == [1.0]
    assert len(store.read_as_of(3.0)) == 4


def test_skew_reports_backward_jump(sub_path: Path) -> None:
    store = Substrate(sub_path)
    store.append(_record(1.0, t_wall=10.0))
    store.append(_record(2.0, t_wall=11.0))
    store.append(_record(1.5, t_wall=11.0))

    report = store.skew()
    assert report["max_backward_jump"] == pytest.approx(0.5)
    assert report["wall_mono_drift"] == pytest.approx(0.5)


def test_shard_rollover(sub_path: Path) -> None:
    store = Substrate(sub_path, shard_size=2)
    for t in (1.0, 2.0, 3.0, 4.0, 5.0):
        store.append(_record(t))

    shards = sorted((sub_path / "shards").glob("shard-*.jsonl"))
    assert len(shards) == 3
    assert [r.seq for r in store.read_all()] == [1, 2, 3, 4, 5]


def test_seq_continues_across_reopen(sub_path: Path) -> None:
    Substrate(sub_path).append(_record(1.0))
    reopened = Substrate(sub_path)
    assert reopened.append(_record(2.0)).seq == 2
    assert [r.seq for r in reopened.read_all()] == [1, 2]


def test_format_mismatch_is_rejected(sub_path: Path) -> None:
    sub_path.mkdir(parents=True, exist_ok=True)
    (sub_path / "FORMAT").write_text("SUBSTRATE-0\n", encoding="utf-8")
    with pytest.raises(FormatError):
        Substrate(sub_path)


def test_seq_must_be_gap_free(sub_path: Path) -> None:
    store = Substrate(sub_path)
    with pytest.raises(RecordError):
        store.append({**_record(1.0), "seq": 7})
