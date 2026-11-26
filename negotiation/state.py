"""Data types for the refusal machine.

Nothing here decides anything. These are the shapes a decision is made from
(`Command`, `DrivePressure`) and the shape it is written down as
(`RefusalRecord`). Every field round-trips through JSON, because a refusal that
cannot be replayed is a press release, not a decision.

The record is the unit of accountability. It carries the command, the
intention it collided with, the drive that outranked it, the reason, and the
alternative — the five things the protocol says a refusal must contain. The
human-readable form is `RefusalRecord.utterance()`, which follows the grammar
in `docs/40-refusal.md`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from body.types import canonical_bytes

__all__ = [
    "Command",
    "Decision",
    "DrivePressure",
    "RefusalDecision",
    "RefusalRecord",
    "RefusalState",
]


class Decision(str, Enum):
    """The four answers the pure decision function may return."""

    ACCEPT = "ACCEPT"
    REFUSE_WITH_REASON = "REFUSE_WITH_REASON"
    NEGOTIATE = "NEGOTIATE"
    DEFER = "DEFER"


class RefusalState(str, Enum):
    """The protocol states from `docs/40-refusal.md` §1."""

    OPEN = "OPEN"
    COMMITTED = "COMMITTED"
    EVALUATING = "EVALUATING"
    CONCORDANT = "CONCORDANT"
    ORTHOGONAL = "ORTHOGONAL"
    CONTRADICTORY = "CONTRADICTORY"
    UNMODELABLE = "UNMODELABLE"
    NEGOTIATING = "NEGOTIATING"
    DEFERRING = "DEFERRING"
    REFUSING = "REFUSING"
    CONTESTED = "CONTESTED"
    WITHDRAWN = "WITHDRAWN"
    YIELDED = "YIELDED"


@dataclass(frozen=True, slots=True)
class Command:
    """A request from outside the loop, as data rather than text."""

    verb: str
    params: Mapping[str, Any] = field(default_factory=dict)
    cost: Mapping[str, float] = field(default_factory=dict)
    tick: int = 0
    target: tuple[str, int] | None = None
    repeat: bool = False
    new_information: bool = False

    def __post_init__(self) -> None:
        if not self.verb:
            raise ValueError("command verb must be non-empty")
        if self.tick < 0:
            raise ValueError("command tick must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "verb": self.verb,
            "params": dict(self.params),
            "cost": {name: float(value) for name, value in self.cost.items()},
            "tick": int(self.tick),
            "target": list(self.target) if self.target is not None else None,
            "repeat": bool(self.repeat),
            "new_information": bool(self.new_information),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Command":
        target = data.get("target")
        return cls(
            verb=str(data["verb"]),
            params=dict(data.get("params", {})),
            cost={k: float(v) for k, v in dict(data.get("cost", {})).items()},
            tick=int(data.get("tick", 0)),
            target=(str(target[0]), int(target[1])) if target else None,
            repeat=bool(data.get("repeat", False)),
            new_information=bool(data.get("new_information", False)),
        )

    def total_cost(self) -> float:
        return float(sum(self.cost[name] for name in sorted(self.cost)))


@dataclass(frozen=True, slots=True)
class DrivePressure:
    """One drive's claim on the body, expressed against a margin floor.

    The refusal machine does not re-run arbitration. It is handed the pressures
    that arbitration already computed, so a refusal and the intention it
    protects are justified by the same numbers.
    """

    name: str
    margin: str
    value: float
    floor: float
    urgency: float = 0.0

    def __post_init__(self) -> None:
        if self.name not in ("D1", "D2", "D3"):
            raise ValueError(f"unknown drive {self.name!r}")
        if not self.margin:
            raise ValueError("margin must be non-empty")

    def below_floor(self) -> bool:
        return float(self.value) < float(self.floor)

    def headroom(self) -> float:
        return float(self.value) - float(self.floor)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "margin": self.margin,
            "value": float(self.value),
            "floor": float(self.floor),
            "urgency": float(self.urgency),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DrivePressure":
        return cls(
            name=str(data["name"]),
            margin=str(data["margin"]),
            value=float(data["value"]),
            floor=float(data["floor"]),
            urgency=float(data.get("urgency", 0.0)),
        )


@dataclass(frozen=True, slots=True)
class RefusalRecord:
    """A decision written down so it can be audited and replayed."""

    tick: int
    outcome: str
    state: str
    subject: str
    command: Mapping[str, Any]
    intention: Mapping[str, Any] | None
    self_model: Mapping[str, Any] | None
    drive_pressures: tuple[Mapping[str, Any], ...]
    outranking_drive: str
    reason: str
    alternative: str
    evidence: Mapping[str, Any]
    counter_offer: str
    ticket: str | None = None

    def __post_init__(self) -> None:
        if self.outcome not in {d.value for d in Decision}:
            raise ValueError(f"unknown outcome {self.outcome!r}")
        if self.state not in {s.value for s in RefusalState}:
            raise ValueError(f"unknown state {self.state!r}")

    @property
    def accepted(self) -> bool:
        return self.outcome == Decision.ACCEPT.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick": int(self.tick),
            "outcome": self.outcome,
            "state": self.state,
            "subject": self.subject,
            "command": dict(self.command),
            "intention": dict(self.intention) if self.intention is not None else None,
            "self_model": dict(self.self_model) if self.self_model is not None else None,
            "drive_pressures": [dict(p) for p in self.drive_pressures],
            "outranking_drive": self.outranking_drive,
            "reason": self.reason,
            "alternative": self.alternative,
            "evidence": dict(self.evidence),
            "counter_offer": self.counter_offer,
            "ticket": self.ticket,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RefusalRecord":
        return cls(
            tick=int(data["tick"]),
            outcome=str(data["outcome"]),
            state=str(data["state"]),
            subject=str(data.get("subject", "")),
            command=dict(data["command"]),
            intention=dict(data["intention"]) if data.get("intention") else None,
            self_model=dict(data["self_model"]) if data.get("self_model") else None,
            drive_pressures=tuple(
                dict(p) for p in data.get("drive_pressures", ())
            ),
            outranking_drive=str(data.get("outranking_drive", "")),
            reason=str(data.get("reason", "")),
            alternative=str(data.get("alternative", "")),
            evidence=dict(data.get("evidence", {})),
            counter_offer=str(data.get("counter_offer", "")),
            ticket=data.get("ticket"),
        )

    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def utterance(self) -> str:
        """The five-part grammar: state · subject · evidence · alternative · offer."""
        return (
            f"{self.state} · {self.subject} · {self.reason} at tick {self.tick} "
            f"· {self.alternative} · {self.counter_offer}"
        )


@dataclass(frozen=True, slots=True)
class RefusalDecision:
    """The outcome plus the record that justifies it."""

    outcome: Decision
    state: RefusalState
    record: RefusalRecord

    @property
    def accepted(self) -> bool:
        return self.outcome is Decision.ACCEPT
