"""The scoring function. Pure, total, and testable against hand arithmetic.

    μ_d(k) = clamp(1 + λ·(k − last_served_d) / T_ref, 1, 3)
    S(g)   = μ_d · ŝ_d · v(g) · (1 + κ·IG(g)) · rev(g) / (1 + c(g))

`c(g)` is the sum of `goal.cost_vector`. There is no randomness, no clock, and
no I/O: the same four inputs always produce the same float.
"""

from __future__ import annotations

from cortex.arbitration.config import ArbitrationConfig
from cortex.drives.state import DriveState
from cortex.goal import Goal

__all__ = ["anti_starvation", "score"]


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def anti_starvation(
    state: DriveState, tick: int, config: ArbitrationConfig
) -> float:
    """Multiplier that grows while a drive goes unserved, capped at 3."""
    elapsed = max(0, int(tick) - int(state.last_served_tick))
    return _clamp(
        1.0 + config.lambda_ * elapsed / config.t_ref,
        1.0,
        3.0,
    )


def score(
    goal: Goal,
    urgency: float,
    state: DriveState,
    tick: int,
    config: ArbitrationConfig,
) -> float:
    """Score one candidate. Higher wins; ties are broken by the caller."""
    mu = anti_starvation(state, tick, config)
    gain = 1.0 + config.kappa * goal.expected_information_gain
    return (
        mu
        * float(urgency)
        * goal.value
        * gain
        * goal.reversibility
        / (1.0 + goal.cost())
    )
