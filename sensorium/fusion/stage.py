"""Fusion: three or more modalities in, one timestamped vector out.

The output is not "vision plus audio". It is a single fixed-width vector whose
channels are named and ordered by `CHANNELS`, with one timestamp and one
freshness per channel. A gridworld and a Pi rig produce the same shape because
they produce the same channel names, not because they produce the same numbers.

Defaults are part of the contract: an absent modality contributes `0.0` with
freshness `0.0` and no mask bit. It never raises and never shifts the vector.

Time is clamped upward. If a bundle arrives with a clock reading older than the
previous fusion, the output keeps the previous timestamp and reports the skew,
because a fused state that travels backwards would make replay ambiguous.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

from body.types import canonical_bytes, modality_bit
from sensorium.fusion.bundle import (
    CHANNELS,
    CHANNEL_MODALITY,
    PerceptionBundle,
    Reading,
)

__all__ = [
    "DEFAULT_FRESHNESS",
    "DEFAULT_VALUE",
    "FusedPerception",
    "FusionStage",
    "fuse",
]

#: Documented defaults for a modality the body cannot supply.
DEFAULT_VALUE = 0.0
DEFAULT_FRESHNESS = 0.0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True, slots=True)
class FusedPerception:
    """One fused, timestamped vector. Same shape on every embodiment."""

    body_id: str
    t_mono: float
    t_wall: float
    seq: int
    values: tuple[float, ...]
    freshness: tuple[float, ...]
    mask: int
    skew: float = 0.0

    def __post_init__(self) -> None:
        if len(self.values) != len(CHANNELS):
            raise ValueError(
                f"vector width {len(self.values)} != {len(CHANNELS)} channels"
            )
        if len(self.freshness) != len(CHANNELS):
            raise ValueError("freshness must align with channels")
        if self.mask < 0:
            raise ValueError("mask must be >= 0")

    @property
    def channels(self) -> tuple[str, ...]:
        return CHANNELS

    @property
    def width(self) -> int:
        return len(self.values)

    def channel(self, name: str) -> float:
        try:
            return self.values[CHANNELS.index(name)]
        except ValueError as exc:
            raise KeyError(f"unknown channel {name!r}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "body_id": self.body_id,
            "t_mono": float(self.t_mono),
            "t_wall": float(self.t_wall),
            "seq": int(self.seq),
            "channels": list(CHANNELS),
            "values": [float(v) for v in self.values],
            "freshness": [float(v) for v in self.freshness],
            "mask": int(self.mask),
            "skew": float(self.skew),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FusedPerception":
        return cls(
            body_id=str(data["body_id"]),
            t_mono=float(data["t_mono"]),
            t_wall=float(data["t_wall"]),
            seq=int(data["seq"]),
            values=tuple(float(v) for v in data["values"]),
            freshness=tuple(float(v) for v in data["freshness"]),
            mask=int(data.get("mask", 0)),
            skew=float(data.get("skew", 0.0)),
        )

    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def fuse(
    bundle: PerceptionBundle,
    previous: FusedPerception | None = None,
) -> FusedPerception:
    """Fuse one bundle into one vector. Pure when `previous` is supplied."""
    readings = dict(bundle.readings())
    raw_t = float(bundle.t_mono)
    t_mono = max(raw_t, previous.t_mono) if previous is not None else raw_t
    skew = t_mono - raw_t

    values: list[float] = []
    freshness: list[float] = []
    mask = 0
    for name in CHANNELS:
        reading: Reading | None = readings.get(name)
        if reading is None or not reading.present:
            values.append(DEFAULT_VALUE)
            freshness.append(DEFAULT_FRESHNESS)
            continue
        values.append(_clamp01(reading.value))
        freshness.append(_clamp01(reading.weight))
        mask |= modality_bit(CHANNEL_MODALITY[name])

    return FusedPerception(
        body_id=str(bundle.body_id),
        t_mono=t_mono,
        t_wall=float(bundle.t_wall),
        seq=int(bundle.seq),
        values=tuple(values),
        freshness=tuple(freshness),
        mask=mask,
        skew=skew,
    )


class FusionStage:
    """A stateful sequence of fusions, for monotonic timestamps and replay."""

    def __init__(self, previous: FusedPerception | None = None) -> None:
        self.previous = previous

    def fuse(self, bundle: PerceptionBundle) -> FusedPerception:
        self.previous = fuse(bundle, self.previous)
        return self.previous

    def state_dict(self) -> dict[str, Any] | None:
        return self.previous.to_dict() if self.previous is not None else None

    @classmethod
    def from_state(cls, state: Mapping[str, Any] | None) -> "FusionStage":
        return cls(FusedPerception.from_dict(state) if state else None)
