"""Arbitration weights and thresholds. Every number lives here.

If a constant affects a decision, it is in `ArbitrationConfig`. Nothing else in
the arbitration layer may hard-code a weight, a threshold, or a window, because
a tunable that is scattered is a tunable that cannot be replayed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

__all__ = ["ArbitrationConfig"]


@dataclass(frozen=True, slots=True)
class ArbitrationConfig:
    """Documented defaults for the whole competition.

    * `lambda_` — anti-starvation rate; urgency grows by this fraction of
      elapsed time over `t_ref`.
    * `t_ref` — ticks after which a starved drive has earned a full `lambda_`.
    * `kappa` — weight of expected information gain in the score.
    * `theta_commit` — minimum score to acquire an intention at all.
    * `theta_preempt` — multiplier the challenger must beat to displace a live
      intention; strictly greater than 1, which is the hysteresis.
    * `refractory_ticks` — damping window after a drive is served.
    * `carry_bonus` — fraction of remaining value credited to a goal that was
      aborted, so nearly-finished work can win the next round.
    * `drift_cost_cap` — maximum total cost of a DRIFT exploration goal.
    """

    lambda_: float = 0.2
    t_ref: int = 40
    kappa: float = 0.5
    theta_commit: float = 0.35
    theta_preempt: float = 1.4
    refractory_ticks: int = 40
    carry_bonus: float = 0.1
    drift_cost_cap: float = 0.02

    def __post_init__(self) -> None:
        if self.t_ref < 1:
            raise ValueError("t_ref must be >= 1")
        if self.theta_preempt <= 1.0:
            raise ValueError("theta_preempt must exceed 1 to create hysteresis")
        if not (0.0 <= self.carry_bonus <= 1.0):
            raise ValueError("carry_bonus must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArbitrationConfig":
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in allowed})
