"""Canonical data types shared by every layer of ULTRON-2030.

Nothing in this module knows about sensors, drives, or storage. These are the
only shapes the agent core is allowed to see: a fused `Snapshot`, the body's
`Margins`, the `Affordance` list a body declares, and the `ActionResult` a body
returns.

Additions beyond the dossier (noted per the build instruction):

* `Margins.FLOORS` and `Margins.worst()` — the viability filter in the drive
  layer needs a single, agreed definition of "which margin is in trouble".
  Per the dossier viability is a minimum, not a mean, so it lives here.
* `Snapshot.canonical_bytes()` / `digest()` — replay and parity demand one
  byte-level definition of a snapshot, owned by the type itself.
* `Cell.modality_mask` is a bitfield; `MODALITIES` fixes the bit order so that
  a mask value means the same thing in every body.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping, Sequence

CANON_VERSION = "snapshot-1"

#: Bit order for `Cell.modality_mask`. A body may support a subset; the bit
#: positions never move.
MODALITIES: tuple[str, ...] = (
    "vision",
    "audio",
    "proximity",
    "thermal",
    "radio",
    "dead_reckoning",
)

CELL_FIELDS: tuple[str, ...] = (
    "frame",
    "slot",
    "value",
    "var",
    "observed_at",
    "freshness",
    "modality_mask",
)

MARGIN_FIELDS: tuple[str, ...] = (
    "power",
    "thermal",
    "compute",
    "storage",
    "sensor_liveness",
)


def modality_bit(name: str) -> int:
    """Return the mask bit for a modality name."""
    if name not in MODALITIES:
        raise ValueError(f"unknown modality {name!r}")
    return 1 << MODALITIES.index(name)


def _clean_float(value: float, digits: int = 6) -> float:
    """Round and normalise a float so canonical bytes are stable.

    `-0.0` is collapsed to `0.0`; without this two equal tensors can produce
    different bytes and replay would report a phantom divergence.
    """
    out = round(float(value), digits)
    return 0.0 if out == 0.0 else out


def _canonicalise(obj: Any) -> Any:
    if isinstance(obj, float):
        return _clean_float(obj)
    if isinstance(obj, Mapping):
        return {str(k): _canonicalise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_canonicalise(v) for v in obj]
    return obj


def canonical_bytes(obj: Any) -> bytes:
    """Serialise a JSON-compatible object to stable, comparison-safe bytes."""
    return json.dumps(
        _canonicalise(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


@dataclass(frozen=True, slots=True)
class Cell:
    """One slot of the fused perceptual tensor.

    A cell is always present. A missing modality leaves a cell with decaying
    `freshness`, frozen `observed_at`, and rising `var`; absence is a value,
    never a hole.
    """

    frame: str
    slot: int
    value: float
    var: float
    observed_at: float
    freshness: float
    modality_mask: int

    def __post_init__(self) -> None:
        if self.slot < 0:
            raise ValueError("slot must be >= 0")
        if self.var < 0.0:
            raise ValueError("var must be >= 0")
        if not (0.0 <= self.freshness <= 1.0):
            raise ValueError(f"freshness out of range: {self.freshness!r}")
        if self.modality_mask < 0:
            raise ValueError("modality_mask must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "slot": int(self.slot),
            "value": float(self.value),
            "var": float(self.var),
            "observed_at": float(self.observed_at),
            "freshness": float(self.freshness),
            "modality_mask": int(self.modality_mask),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Cell":
        missing = [f for f in CELL_FIELDS if f not in data]
        if missing:
            raise ValueError(f"cell missing fields: {missing}")
        return cls(
            frame=str(data["frame"]),
            slot=int(data["slot"]),
            value=float(data["value"]),
            var=float(data["var"]),
            observed_at=float(data["observed_at"]),
            freshness=float(data["freshness"]),
            modality_mask=int(data["modality_mask"]),
        )


@dataclass(frozen=True, slots=True)
class Snapshot:
    """One timestamped, fused perceptual state — the agent's entire world.

    Two clocks travel together: `t_mono` orders events causally, `t_wall` is
    what the agent may say out loud. `residual` is the prediction error for
    each cell, aligned by index with `cells`; the snapshot carries its own
    surprise instead of making downstream layers recompute it.
    """

    t_mono: float
    t_wall: float
    seq: int
    body_id: str
    cells: tuple[Cell, ...]
    residual: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.seq < 0:
            raise ValueError("seq must be >= 0")
        if len(self.cells) != len(self.residual):
            raise ValueError(
                "residual must align with cells: "
                f"{len(self.cells)} cells vs {len(self.residual)} residuals"
            )
        if not self.body_id:
            raise ValueError("body_id must be non-empty")

    @property
    def width(self) -> int:
        return len(self.cells)

    def slot_index(self) -> dict[tuple[str, int], int]:
        return {(c.frame, c.slot): i for i, c in enumerate(self.cells)}

    def cell(self, frame: str, slot: int) -> Cell:
        for c in self.cells:
            if c.frame == frame and c.slot == slot:
                return c
        raise KeyError(f"no cell {frame!r}/{slot}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "t_mono": float(self.t_mono),
            "t_wall": float(self.t_wall),
            "seq": int(self.seq),
            "body_id": self.body_id,
            "cells": [c.to_dict() for c in self.cells],
            "residual": [float(r) for r in self.residual],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Snapshot":
        for field in ("t_mono", "t_wall", "seq", "body_id", "cells", "residual"):
            if field not in data:
                raise ValueError(f"snapshot missing field {field!r}")
        return cls(
            t_mono=float(data["t_mono"]),
            t_wall=float(data["t_wall"]),
            seq=int(data["seq"]),
            body_id=str(data["body_id"]),
            cells=tuple(Cell.from_dict(c) for c in data["cells"]),
            residual=tuple(float(r) for r in data["residual"]),
        )

    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class Margins:
    """Remaining headroom on each axis the body must stay alive on.

    All values are `[0, 1]`, and higher is always better: `1.0` is untouched,
    `0.0` is dead. The drive layer reads `worst()` because viability is a
    minimum, not a mean.
    """

    power: float
    thermal: float
    compute: float
    storage: float
    sensor_liveness: float

    #: Below these, the homeostatic drive enters VITAL. Sensor liveness has a
    #: high floor on purpose: a body that cannot sense is already dead.
    FLOORS: ClassVar[dict[str, float]] = {
        "power": 0.25,
        "thermal": 0.20,
        "compute": 0.15,
        "storage": 0.10,
        "sensor_liveness": 0.90,
    }

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in MARGIN_FIELDS}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Margins":
        missing = [f for f in MARGIN_FIELDS if f not in data]
        if missing:
            raise ValueError(f"margins missing fields: {missing}")
        return cls(**{name: float(data[name]) for name in MARGIN_FIELDS})

    def worst(self) -> tuple[str, float]:
        """Return `(name, value)` for the margin closest to failure."""
        name = min(MARGIN_FIELDS, key=lambda n: getattr(self, n))
        return name, float(getattr(self, name))

    def below_floor(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in MARGIN_FIELDS
            if getattr(self, name) < self.FLOORS[name]
        )


@dataclass(frozen=True, slots=True)
class Affordance:
    """A verb this body can actually perform, plus its parameter names.

    An affordance is a claim the body makes about itself. The core treats the
    list as authoritative: no goal may name a verb that is not here.
    """

    verb: str
    params: tuple[str, ...]
    cost: Mapping[str, float]
    description: str

    def __post_init__(self) -> None:
        if not self.verb:
            raise ValueError("verb must be non-empty")
        if self.verb != self.verb.upper():
            raise ValueError(f"verb must be upper-case: {self.verb!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "verb": self.verb,
            "params": list(self.params),
            "cost": {k: float(v) for k, v in self.cost.items()},
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class ActionResult:
    """What the body reports after an attempt, including what it cost."""

    ok: bool
    verb: str
    params: Mapping[str, Any]
    t_mono: float
    detail: str
    margin_delta: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "verb": self.verb,
            "params": dict(self.params),
            "t_mono": float(self.t_mono),
            "detail": self.detail,
            "margin_delta": {k: float(v) for k, v in self.margin_delta.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActionResult":
        return cls(
            ok=bool(data["ok"]),
            verb=str(data["verb"]),
            params=dict(data["params"]),
            t_mono=float(data["t_mono"]),
            detail=str(data["detail"]),
            margin_delta=dict(data["margin_delta"]),
        )


__all__ = [
    "Affordance",
    "ActionResult",
    "CANON_VERSION",
    "CELL_FIELDS",
    "Cell",
    "MARGIN_FIELDS",
    "MODALITIES",
    "Margins",
    "Snapshot",
    "canonical_bytes",
    "modality_bit",
]
