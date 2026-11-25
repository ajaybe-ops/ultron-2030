"""D3 — the legibility drive. Wants the other's model of the agent to be accurate.

    ŝ_S = clamp(w_p·P + w_d·D̂)        w_p=0.4, w_d=0.6

This drive is deliberately not "be near the human". It is self-interested: a
mispredicted agent gets interrupted, and interruptions are expensive. Presence
`P` is sensed, never inferred from a chat event. The gridworld has no presence
modality, so `P = 0.0` and the drive is quiet but still computes — which is the
point. The loop is complete even when a drive has nothing to say.

`D̂` is repair debt, read from `DriveState.deficit`. In this build there is no
dialogue, so nothing can create debt and it stays 0.0. It is wired now so the
refusal step has somewhere to write; it is not silently stubbed.

Candidates (DISCLOSE, APPROACH, NEGOTIATE) are all emitted and all filtered by
the gridworld's affordances. Zero candidates survive today. That is expected.
"""

from __future__ import annotations

from body.types import Margins, Snapshot
from cortex.drives.state import DriveState
from cortex.goal import Goal

__all__ = ["PRESENCE_THRESHOLD", "candidates", "presence", "urgency"]

PRESENCE_THRESHOLD = 0.5
WEIGHTS = {"presence": 0.4, "repair": 0.6}

_COSTS = {
    "DISCLOSE": {"compute": 0.002},
    "APPROACH": {"power": 0.006},
    "NEGOTIATE": {"compute": 0.004},
}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def presence(snapshot: Snapshot) -> float:
    """Confidence that another agent is present, read from presence slots.

    A body with no `presence` frame reports 0.0. Nothing is inferred from
    conversation, because the drive runs when there is no conversation.
    """
    slots = [c for c in snapshot.cells if c.frame == "presence"]
    if not slots:
        return 0.0
    return float(sum(c.freshness for c in slots) / len(slots))


def urgency(
    snapshot: Snapshot,
    margins: Margins,
    ledger: object,
    state: DriveState,
) -> float:
    """Compute D3 urgency from sensed presence and accumulated repair debt."""
    p = presence(snapshot)
    debt = _clamp(state.deficit)
    return _clamp(WEIGHTS["presence"] * p + WEIGHTS["repair"] * debt)


def candidates(
    snapshot: Snapshot,
    margins: Margins,
    ledger: object,
    state: DriveState,
) -> list[Goal]:
    """Emit the three social verbs; the affordance filter decides their fate."""
    p = presence(snapshot)
    debt = _clamp(state.deficit)
    values = {
        "DISCLOSE": _clamp(0.3 + 0.7 * p),
        "APPROACH": _clamp(p),
        "NEGOTIATE": _clamp(debt),
    }
    return [
        Goal(
            verb=verb,
            drive="D3",
            value=values[verb],
            cost_vector=dict(_COSTS[verb]),
            reversibility=1.0,
            latency=1,
            expected_information_gain=0.0,
        )
        for verb in ("DISCLOSE", "APPROACH", "NEGOTIATE")
    ]
