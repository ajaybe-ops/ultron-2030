"""D2 — the homeostatic drive. Wants the body to remain viable.

    ŝ_H = 1 − min(margins)

Viability is a minimum, never a mean, so this drive reads `Margins.worst()`.
Floors come from `Margins.FLOORS`; when any margin is below its floor the drive
enters VITAL, which is a mode the engine acts on, and leaves a deficit of 1.0
behind so the state record carries the event forward.

Candidates are emitted for all five verbs the dossier names. The drive does not
know what body it is in and must not: the affordance filter removes the verbs
the body does not declare. In the gridworld only REST survives, and that is the
correct outcome, not a shortcut.
"""

from __future__ import annotations

from body.types import Margins, Snapshot
from cortex.drives.state import DriveState
from cortex.goal import Goal

__all__ = ["candidates", "urgency"]

_VERB_COSTS = {
    "REST": {},
    "SHED_LOAD": {"compute": 0.05},
    "SEEK_CHARGE": {"power": 0.10},
    "COOL": {"thermal": 0.10},
    "REACQUIRE_SENSOR": {"compute": 0.02},
}

_LATENCY = {
    "REST": 3,
    "SHED_LOAD": 1,
    "SEEK_CHARGE": 4,
    "COOL": 2,
    "REACQUIRE_SENSOR": 2,
}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def urgency(
    snapshot: Snapshot,
    margins: Margins,
    ledger: object,
    state: DriveState,
) -> float:
    """Compute D2 urgency and latch VITAL when any margin is under its floor."""
    _, worst = margins.worst()
    below = margins.below_floor()
    state.vital = bool(below)
    if state.vital:
        state.deficit = 1.0
    return _clamp(1.0 - worst)


def candidates(
    snapshot: Snapshot,
    margins: Margins,
    ledger: object,
    state: DriveState,
) -> list[Goal]:
    """Emit one goal per viability axis; the filter keeps what the body can do."""
    values = {
        "REST": 1.0 - margins.worst()[1],
        "SHED_LOAD": 1.0 - margins.compute,
        "SEEK_CHARGE": 1.0 - margins.power,
        "COOL": 1.0 - margins.thermal,
        "REACQUIRE_SENSOR": 1.0 - margins.sensor_liveness,
    }
    out: list[Goal] = []
    for verb in ("REST", "SHED_LOAD", "SEEK_CHARGE", "COOL", "REACQUIRE_SENSOR"):
        out.append(
            Goal(
                verb=verb,
                drive="D2",
                value=_clamp(values[verb]),
                cost_vector=dict(_VERB_COSTS[verb]),
                reversibility=1.0,
                latency=_LATENCY[verb],
                expected_information_gain=0.0,
            )
        )
    return out
