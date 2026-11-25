"""The Goal: a candidate intention before it is leased.

A `Goal` is what a drive emits and what arbitration scores. It is pure data so
it can be scored, compared, serialised into an intention, and replayed. The
verb must be one the body declares; that is enforced later by the affordance
filter, not here, because a drive is allowed to want something the body cannot
do — wanting is not acting.

`target` addresses a cell as `(frame, slot)`. It is `None` for whole-body verbs
such as REST. `cost_vector` uses the same axis names as `Margins`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

__all__ = ["Goal", "Target"]

Target = tuple[str, int]


@dataclass(frozen=True, slots=True)
class Goal:
    """One candidate the arbitration layer may commit to."""

    verb: str
    drive: str
    value: float
    cost_vector: Mapping[str, float]
    reversibility: float
    latency: int
    expected_information_gain: float
    target: Target | None = None
    deadline: int | None = None

    def __post_init__(self) -> None:
        if not self.verb:
            raise ValueError("verb must be non-empty")
        if self.drive not in ("D1", "D2", "D3"):
            raise ValueError(f"unknown drive {self.drive!r}")
        if not (0.0 <= self.value <= 1.0):
            raise ValueError(f"value out of range: {self.value!r}")
        if not (0.0 < self.reversibility <= 1.0):
            raise ValueError(f"reversibility out of range: {self.reversibility!r}")
        if self.latency < 1:
            raise ValueError("latency must be >= 1 tick")
        if not (0.0 <= self.expected_information_gain <= 1.0):
            raise ValueError("expected_information_gain must be in [0, 1]")
        for amount in self.cost_vector.values():
            if amount < 0.0:
                raise ValueError("cost_vector entries must be non-negative")

    def cost(self) -> float:
        """Total cost in margin units; the denominator of the score.

        Summation is over sorted axis names so the result does not depend on
        the insertion order of `cost_vector`. A goal rebuilt from JSON has its
        keys in sorted order; a goal built by a drive has them in declaration
        order. Floating-point addition is not associative, so summing in a
        fixed order is what keeps replay byte-identical.
        """
        return float(sum(self.cost_vector[name] for name in sorted(self.cost_vector)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "verb": self.verb,
            "drive": self.drive,
            "value": float(self.value),
            "cost_vector": {k: float(v) for k, v in self.cost_vector.items()},
            "reversibility": float(self.reversibility),
            "latency": int(self.latency),
            "expected_information_gain": float(self.expected_information_gain),
            "target": list(self.target) if self.target is not None else None,
            "deadline": self.deadline,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Goal":
        target = data.get("target")
        return cls(
            verb=str(data["verb"]),
            drive=str(data["drive"]),
            value=float(data["value"]),
            cost_vector={k: float(v) for k, v in data["cost_vector"].items()},
            reversibility=float(data["reversibility"]),
            latency=int(data["latency"]),
            expected_information_gain=float(data["expected_information_gain"]),
            target=(str(target[0]), int(target[1])) if target else None,
            deadline=int(data["deadline"]) if data.get("deadline") is not None else None,
        )
