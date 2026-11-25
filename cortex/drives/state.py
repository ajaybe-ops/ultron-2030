"""Persistent state for one intrinsic drive.

Every drive keeps exactly this record between ticks. It is small on purpose:
the engine may hold only `config`, `drive_states`, and `current`, so anything a
drive needs to remember has to live here and has to serialise, or replay is
lying.

Two fields go beyond the dossier's minimum and are documented here because they
are required to implement it:

* `vital` — the homeostatic mode. The dossier calls VITAL "a mode, not a bigger
  number"; a mode needs a flag.
* `residual_ring` — the fixed-size history D1 needs to compute a running 95th
  percentile. It is bounded at `RING_SIZE` samples so state never grows without
  limit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

__all__ = ["RING_SIZE", "DriveState"]

RING_SIZE = 64


@dataclass
class DriveState:
    """Per-drive memory: service history, refractory window, and deficit."""

    last_served_tick: int = 0
    refractory_until: int = 0
    deficit: float = 0.0
    served_count: int = 0
    vital: bool = False
    residual_ring: tuple[float, ...] = field(default_factory=tuple)

    def push_residual(self, value: float) -> None:
        """Append one sample, keeping the newest `RING_SIZE` values."""
        ring = self.residual_ring + (float(value),)
        self.residual_ring = ring[-RING_SIZE:]

    def to_dict(self) -> dict[str, Any]:
        return {
            "last_served_tick": int(self.last_served_tick),
            "refractory_until": int(self.refractory_until),
            "deficit": float(self.deficit),
            "served_count": int(self.served_count),
            "vital": bool(self.vital),
            "residual_ring": [float(v) for v in self.residual_ring],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DriveState":
        return cls(
            last_served_tick=int(data.get("last_served_tick", 0)),
            refractory_until=int(data.get("refractory_until", 0)),
            deficit=float(data.get("deficit", 0.0)),
            served_count=int(data.get("served_count", 0)),
            vital=bool(data.get("vital", False)),
            residual_ring=tuple(float(v) for v in data.get("residual_ring", ())),
        )


def fresh_states() -> dict[str, DriveState]:
    """A new, independent state record for each drive, in fixed order."""
    return {"D1": DriveState(), "D2": DriveState(), "D3": DriveState()}
