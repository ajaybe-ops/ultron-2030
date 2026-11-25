"""SUBSTRATE-1: the append-only, content-addressed belief store.

This is not a database. It is a log of what the agent perceived, inferred, and
did, ordered by monotonic time, never mutated in place, and readable as of any
past moment. "Updating" a belief never overwrites: it appends a new record and,
at compaction, writes a new stratum whose provenance points back at the exact
`seq` values it summarises. Nothing is ever deleted, because a self that cannot
reconstruct its own past cannot be debugged and cannot learn from failure.

File layout::

    <root>/FORMAT                       # b"SUBSTRATE-1\\n"
    <root>/shards/shard-000000.jsonl    # append shard, one JSON record per line
    <root>/shards/compact-000000.jsonl  # compaction stratum
    <root>/shards/<name>.superseded.json# supersession marker for <name>

Crash-safety rule: appends are one `write()` of `<json>\\n` followed by
`fsync`. A torn tail therefore lacks its trailing newline (or fails to parse),
and is ignored on read. A torn line anywhere other than the final shard is
corruption and raises: only the tail of the log can be interrupted.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = [
    "FORMAT",
    "CorruptionError",
    "FormatError",
    "Record",
    "RecordError",
    "Substrate",
    "SubstrateError",
]

FORMAT = "SUBSTRATE-1"
FORMAT_FILE = "FORMAT"
SHARD_DIR = "shards"
SUPERSEDED_SUFFIX = ".superseded.json"

REQUIRED_FIELDS: tuple[str, ...] = (
    "seq",
    "t_mono",
    "t_wall",
    "kind",
    "provenance",
    "payload",
)


class SubstrateError(RuntimeError):
    """Base class for every substrate failure."""


class FormatError(SubstrateError):
    """On-disk format string is missing or does not match this reader."""


class RecordError(SubstrateError):
    """A record is malformed or violates append ordering."""


class CorruptionError(SubstrateError):
    """A shard is damaged somewhere other than its tail."""


@dataclass(frozen=True, slots=True)
class Record:
    """One immutable entry in the log.

    `provenance` answers "why does this exist": which snapshot, which rule,
    which earlier `seq` values. `payload` is whatever the writer needed to
    keep. Neither is interpreted by the store.
    """

    seq: int
    t_mono: float
    t_wall: float
    kind: str
    provenance: dict[str, Any]
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": int(self.seq),
            "t_mono": float(self.t_mono),
            "t_wall": float(self.t_wall),
            "kind": str(self.kind),
            "provenance": copy.deepcopy(self.provenance),
            "payload": copy.deepcopy(self.payload),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Record":
        missing = [f for f in REQUIRED_FIELDS if f not in data]
        if missing:
            raise RecordError(f"record missing fields: {missing}")
        if not isinstance(data["provenance"], Mapping):
            raise RecordError("provenance must be a mapping")
        if not isinstance(data["payload"], Mapping):
            raise RecordError("payload must be a mapping")
        kind = str(data["kind"])
        if not kind:
            raise RecordError("kind must be non-empty")
        seq = int(data["seq"])
        if seq < 0:
            raise RecordError("seq must be >= 0")
        return cls(
            seq=seq,
            t_mono=float(data["t_mono"]),
            t_wall=float(data["t_wall"]),
            kind=kind,
            provenance=dict(data["provenance"]),
            payload=dict(data["payload"]),
        )

    def encode(self) -> bytes:
        return (
            json.dumps(
                self.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )


class Substrate:
    """Append-only belief log with as-of reads and non-destructive compaction.

    `read_as_of(t)` replays the log in `seq` order and folds non-monotonic
    clock readings upward (`eff = max(eff, t_mono)`). A backwards clock jump
    therefore collapses into the previous instant instead of reordering
    history or dropping records. Use `skew()` to see how much the clock lied.
    """

    def __init__(self, root: str | os.PathLike[str], shard_size: int = 1000) -> None:
        if shard_size < 1:
            raise ValueError("shard_size must be >= 1")
        self.root = Path(root)
        self.shard_size = int(shard_size)
        self._shard_dir = self.root / SHARD_DIR
        self._shard_dir.mkdir(parents=True, exist_ok=True)
        self._check_format()
        self._next_seq = 1
        self._active_name = ""
        self._active_count = 0
        self._recover()

    # ------------------------------------------------------------------ setup

    def _check_format(self) -> None:
        path = self.root / FORMAT_FILE
        if path.exists():
            found = path.read_text(encoding="utf-8").strip()
            if found != FORMAT:
                raise FormatError(
                    f"{self.root} is {found!r}, this reader speaks {FORMAT!r}"
                )
            return
        path.write_text(FORMAT + "\n", encoding="utf-8")
        _fsync_path(path)

    def _recover(self) -> None:
        records, torn = self._scan()
        if torn:
            self._truncate_torn_tail()
        self._next_seq = (max((r.seq for r in records), default=0)) + 1
        shards = self._shard_paths()
        append_shards = [p for p in shards if p.name.startswith("shard-")]
        if append_shards:
            self._active_name = append_shards[-1].name
            self._active_count = len(self._read_shard(append_shards[-1])[0])
        else:
            self._active_name = "shard-000000.jsonl"
            self._active_count = 0

    def _truncate_torn_tail(self) -> None:
        """Drop an incomplete final write so the next append starts clean.

        Only unacknowledged bytes are removed: no record that `_scan` could
        parse is touched, and no shard file disappears.
        """
        paths = self._shard_paths()
        if not paths:
            return
        path = paths[-1]
        raw = path.read_bytes()
        boundary = raw.rfind(b"\n")
        keep = boundary + 1 if boundary >= 0 else 0
        with path.open("r+b") as handle:
            handle.truncate(keep)
            handle.flush()
            os.fsync(handle.fileno())

    # ------------------------------------------------------------------ paths

    def _shard_paths(self) -> list[Path]:
        return sorted(p for p in self._shard_dir.glob("*.jsonl"))

    def _marker_paths(self) -> list[Path]:
        return sorted(p for p in self._shard_dir.glob("*" + SUPERSEDED_SUFFIX))

    def _active_path(self) -> Path:
        return self._shard_dir / self._active_name

    # ------------------------------------------------------------------ write

    def append(self, record: Record | Mapping[str, Any]) -> Record:
        """Append one record and fsync it.

        The caller may omit `seq`; the store assigns the next one. An explicit
        `seq` is honoured only if it is exactly the next one, which keeps the
        log gap-free and order-preserving.
        """
        if isinstance(record, Record):
            data = record.to_dict()
        elif isinstance(record, Mapping):
            data = copy.deepcopy(dict(record))
        else:
            raise RecordError("append() takes a Record or a mapping")

        if int(data.get("seq", self._next_seq)) != self._next_seq:
            raise RecordError(
                f"seq must be {self._next_seq}, got {data.get('seq')!r}; "
                "the log is gap-free"
            )
        data["seq"] = self._next_seq

        if self._active_count >= self.shard_size:
            self._roll_over()

        provenance = data.get("provenance")
        if not isinstance(provenance, Mapping):
            raise RecordError("provenance must be a mapping")
        provenance = dict(provenance)
        provenance.setdefault("shard", self._active_name)
        data["provenance"] = provenance

        stored = Record.from_dict(data)
        path = self._active_path()
        line = stored.encode()
        with path.open("ab") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

        self._next_seq += 1
        self._active_count += 1
        return stored

    def _roll_over(self) -> None:
        shards = [p for p in self._shard_paths() if p.name.startswith("shard-")]
        index = len(shards)
        self._active_name = f"shard-{index:06d}.jsonl"
        self._active_count = 0
        path = self._active_path()
        path.touch(exist_ok=True)
        _fsync_path(path)

    # ------------------------------------------------------------------- read

    def _read_shard(self, path: Path) -> tuple[list[Record], bool]:
        """Return `(records, torn_tail)` for one shard file."""
        raw = path.read_bytes()
        if not raw:
            return [], False
        chunks = raw.split(b"\n")
        tail = chunks.pop()
        torn = bool(tail.strip())
        records: list[Record] = []
        for chunk in chunks:
            if not chunk.strip():
                continue
            records.append(Record.from_dict(json.loads(chunk.decode("utf-8"))))
        return records, torn

    def _scan(self) -> tuple[list[Record], bool]:
        """Read every shard in order, tolerating a torn final line only at the end."""
        paths = self._shard_paths()
        out: list[Record] = []
        torn_seen = False
        for position, path in enumerate(paths):
            try:
                records, torn = self._read_shard(path)
            except json.JSONDecodeError as exc:
                raise CorruptionError(f"{path.name} is not valid JSONL") from exc
            if torn:
                if position != len(paths) - 1:
                    raise CorruptionError(
                        f"{path.name} has a torn tail but is not the final shard"
                    )
                torn_seen = True
            out.extend(records)
        out.sort(key=lambda r: r.seq)
        return out, torn_seen

    def read_all(self) -> list[Record]:
        """Every record ever written, superseded or not, in `seq` order."""
        return self._scan()[0]

    def read_as_of(self, t_mono: float) -> list[Record]:
        """The log as it stood at monotonic instant `t_mono`.

        Never reads "latest". Compaction later in time does not change this
        answer, which is what makes past beliefs reconstructable after the
        agent has learned better.
        """
        out: list[Record] = []
        effective = float("-inf")
        for record in self.read_all():
            effective = max(effective, record.t_mono)
            if effective > float(t_mono) + 1e-9:
                break
            out.append(record)
        return out

    def _superseded(self) -> tuple[set[int], set[str]]:
        seqs: set[int] = set()
        shards: set[str] = set()
        for marker in self._marker_paths():
            data = json.loads(marker.read_text(encoding="utf-8"))
            seqs.update(int(s) for s in data.get("superseded_seqs", ()))
            if data.get("shard_fully_superseded"):
                shards.add(str(data.get("shard", "")))
        return seqs, shards

    def current(self) -> list[Record]:
        """The live view: every record not superseded by a later stratum."""
        seqs, shards = self._superseded()
        return [
            r
            for r in self.read_all()
            if r.seq not in seqs and self._shard_of(r) not in shards
        ]

    def _shard_of(self, record: Record) -> str:
        tokens = record.provenance.get("shard")
        return str(tokens) if tokens else ""

    def beliefs_as_of(self, t_mono: float) -> dict[str, Record]:
        """Fold belief records by `payload["key"]`, last write winning."""
        folded: dict[str, Record] = {}
        for record in self.read_as_of(t_mono):
            if record.kind != "belief":
                continue
            folded[_belief_key(record)] = record
        return folded

    def skew(self) -> dict[str, float]:
        """Summarise how badly the wall and monotonic clocks disagree."""
        records = self.read_all()
        backward = 0.0
        drift = 0.0
        previous: Record | None = None
        for record in records:
            if previous is not None:
                backward = max(backward, previous.t_mono - record.t_mono)
                wall_step = record.t_wall - previous.t_wall
                mono_step = record.t_mono - previous.t_mono
                drift += abs(wall_step - mono_step)
            previous = record
        return {"max_backward_jump": backward, "wall_mono_drift": drift}

    # -------------------------------------------------------------- compaction

    def compact(self, kind: str = "belief") -> list[Record]:
        """Write surviving beliefs of `kind` into a new stratum.

        Survivors are the last record per `payload["key"]` among records not
        already superseded. Each survivor's provenance gains
        `derived_from: [seq, ...]` naming every original it stands for, and
        `shard: <new stratum>`. Source shards get a marker; no file is removed.
        """
        if not kind:
            raise ValueError("kind must be non-empty")

        superseded_seqs, _ = self._superseded()
        candidates = [
            r
            for r in self.read_all()
            if r.kind == kind and r.seq not in superseded_seqs
        ]
        if not candidates:
            return []

        survivors: dict[str, Record] = {}
        for record in candidates:
            survivors[_belief_key(record)] = record

        t_mono = max((r.t_mono for r in self.read_all()), default=0.0) + 1e-6
        t_wall = max((r.t_wall for r in self.read_all()), default=0.0)
        stratum = len(list(self._shard_dir.glob("compact-*.jsonl"))) + 1
        name = f"compact-{stratum - 1:06d}.jsonl"

        written: list[Record] = []
        for offset, survivor in enumerate(sorted(survivors.values(), key=lambda r: r.seq)):
            seqs = sorted(
                r.seq
                for r in candidates
                if _belief_key(r) == _belief_key(survivor)
            )
            provenance = copy.deepcopy(survivor.provenance)
            provenance.update(
                {
                    "rule": "compaction",
                    "stratum": stratum,
                    "derived_from": seqs,
                    "shard": name,
                }
            )
            written.append(
                Record(
                    seq=self._next_seq,
                    t_mono=t_mono + offset * 1e-9,
                    t_wall=t_wall,
                    kind=kind,
                    provenance=provenance,
                    payload=copy.deepcopy(survivor.payload),
                )
            )
            self._next_seq += 1

        path = self._shard_dir / name
        with path.open("wb") as handle:
            for record in written:
                handle.write(record.encode())
            handle.flush()
            os.fsync(handle.fileno())

        self._mark_superseded(name, stratum, written[0].seq, written[-1].seq, candidates)
        return written

    def _mark_superseded(
        self,
        new_name: str,
        stratum: int,
        first_seq: int,
        last_seq: int,
        superseding: Sequence[Record],
    ) -> None:
        by_shard: dict[str, list[int]] = {}
        for record in superseding:
            by_shard.setdefault(self._shard_of(record), []).append(int(record.seq))

        for shard_name, seqs in by_shard.items():
            if not shard_name:
                continue
            marker = self._shard_dir / (shard_name + SUPERSEDED_SUFFIX)
            payload: dict[str, Any] = {
                "format": FORMAT,
                "shard": shard_name,
                "superseded_by": new_name,
                "superseded_seqs": sorted(set(seqs)),
                "shard_fully_superseded": False,
                "stratum": stratum,
                "at_seq": last_seq,
                "first_seq": first_seq,
            }
            if marker.exists():
                previous = json.loads(marker.read_text(encoding="utf-8"))
                merged = sorted(
                    set(previous.get("superseded_seqs", ())) | set(payload["superseded_seqs"])
                )
                payload["superseded_seqs"] = merged
                payload["superseded_by"] = new_name
            source = self._shard_dir / shard_name
            if source.exists():
                total = len(self._read_shard(source)[0])
                payload["shard_fully_superseded"] = (
                    len(payload["superseded_seqs"]) >= total
                )
            marker.write_text(
                json.dumps(payload, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            _fsync_path(marker)


def _belief_key(record: Record) -> str:
    key = record.payload.get("key")
    if isinstance(key, str) and key:
        return key
    return f"seq:{record.seq}"


def _fsync_path(path: Path) -> None:
    """Best-effort durability for a directory entry or a fresh file.

    Platforms without directory fsync (notably Windows) simply skip it; the
    record-level `fsync` on the append handle is the guarantee that matters.
    """
    try:
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    except OSError:
        return
