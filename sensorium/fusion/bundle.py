"""Perception bundles: the only thing the fusion stage is allowed to see.

A bundle is a timestamped set of modality readings from one body. It is not a
camera, a microphone, or a PIR driver. The gridworld adapter reads a `Snapshot`
and `Margins`; the Pi adapter reads camera frames, PIR events, and microphone
energy. Both produce the same five named channels, in the same order, with the
same defaults, which is what makes fusion embodiment-agnostic.

Missing is a value. A modality the body cannot supply is simply absent from
`readings()`; fusion fills it with `0.0` and freshness `0.0`. Nothing raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from body.types import Margins, Snapshot

__all__ = [
    "CHANNELS",
    "CHANNEL_MODALITY",
    "GridWorldBundle",
    "PerceptionBundle",
    "PiPerceptionBundle",
    "Reading",
    "StaticBundle",
]

#: Fixed channel order. This is the vector shape every embodiment must meet.
CHANNELS: tuple[str, ...] = (
    "occupancy",
    "surprise",
    "viability",
    "presence",
    "audio",
)

#: Which `body.types.MODALITIES` bit a channel contributes to the mask.
CHANNEL_MODALITY: dict[str, str] = {
    "occupancy": "vision",
    "surprise": "dead_reckoning",
    "viability": "thermal",
    "presence": "proximity",
    "audio": "audio",
}


@dataclass(frozen=True, slots=True)
class Reading:
    """One modality's contribution, already reduced to a scalar."""

    value: float
    weight: float = 1.0
    observed_at: float = 0.0
    present: bool = True

    def __post_init__(self) -> None:
        if not (0.0 <= self.weight <= 1.0):
            raise ValueError(f"weight out of range: {self.weight!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": float(self.value),
            "weight": float(self.weight),
            "observed_at": float(self.observed_at),
            "present": bool(self.present),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Reading":
        return cls(
            value=float(data["value"]),
            weight=float(data.get("weight", 1.0)),
            observed_at=float(data.get("observed_at", 0.0)),
            present=bool(data.get("present", True)),
        )


@runtime_checkable
class PerceptionBundle(Protocol):
    """What fusion needs from any embodiment."""

    body_id: str
    t_mono: float
    t_wall: float
    seq: int

    def readings(self) -> Mapping[str, Reading]:
        """Named modality readings; omit what this body cannot sense."""


@dataclass(frozen=True, slots=True)
class StaticBundle:
    """A concrete bundle, used by tests and by both adapters."""

    body_id: str
    t_mono: float
    t_wall: float
    seq: int
    data: Mapping[str, Reading] = field(default_factory=dict)

    def readings(self) -> Mapping[str, Reading]:
        return dict(self.data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "body_id": self.body_id,
            "t_mono": float(self.t_mono),
            "t_wall": float(self.t_wall),
            "seq": int(self.seq),
            "readings": {
                name: self.data[name].to_dict() for name in sorted(self.data)
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StaticBundle":
        return cls(
            body_id=str(data["body_id"]),
            t_mono=float(data["t_mono"]),
            t_wall=float(data["t_wall"]),
            seq=int(data["seq"]),
            data={
                str(name): Reading.from_dict(reading)
                for name, reading in dict(data.get("readings", {})).items()
            },
        )


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class GridWorldBundle(StaticBundle):
    """Gridworld adapter: occupancy, surprise, viability, proximity, no audio."""

    @classmethod
    def from_snapshot(cls, snapshot: Snapshot, margins: Margins) -> "GridWorldBundle":
        vision = [
            float(cell.value)
            for cell in snapshot.cells
            if cell.frame == "grid" and cell.freshness > 0.0
        ]
        if not vision:
            vision = [
                float(cell.value) for cell in snapshot.cells if cell.frame == "grid"
            ]
        proximity = next(
            (float(cell.value) for cell in snapshot.cells if cell.frame == "proximity"),
            0.0,
        )
        observed_at = float(snapshot.t_mono)
        return cls(
            body_id=snapshot.body_id,
            t_mono=float(snapshot.t_mono),
            t_wall=float(snapshot.t_wall),
            seq=int(snapshot.seq),
            data={
                "occupancy": Reading(_mean(vision), 1.0, observed_at),
                "surprise": Reading(
                    _mean([abs(float(r)) for r in snapshot.residual]), 1.0, observed_at
                ),
                "viability": Reading(
                    min(margins.as_dict().values()), 1.0, observed_at
                ),
                "presence": Reading(proximity, 1.0, observed_at),
            },
        )


class PiPerceptionBundle(StaticBundle):
    """Pi adapter: camera frames, PIR events, microphone energy.

    The physical rig body is Phase 3. This adapter exists now so the fusion
    contract can be tested against the hardware's modality set without a Pi.
    """

    @classmethod
    def from_frames(
        cls,
        camera: Sequence[Sequence[float]] | Sequence[float],
        pir: float,
        mic_energy: float,
        margins: Margins,
        residual: Sequence[float] = (),
        *,
        body_id: str = "rig-pi-0",
        t_mono: float = 0.0,
        t_wall: float = 0.0,
        seq: int = 0,
    ) -> "PiPerceptionBundle":
        pixels: list[float] = []
        for row in camera:
            if isinstance(row, (int, float)):
                pixels.append(float(row))
            else:
                pixels.extend(float(value) for value in row)
        observed_at = float(t_mono)
        return cls(
            body_id=body_id,
            t_mono=float(t_mono),
            t_wall=float(t_wall),
            seq=int(seq),
            data={
                "occupancy": Reading(_mean(pixels), 1.0, observed_at),
                "surprise": Reading(
                    _mean([abs(float(r)) for r in residual]), 1.0, observed_at
                ),
                "viability": Reading(
                    min(margins.as_dict().values()), 1.0, observed_at
                ),
                "presence": Reading(_clamp01(pir), 1.0, observed_at),
                "audio": Reading(_clamp01(mic_energy), 1.0, observed_at),
            },
        )
