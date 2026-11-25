"""The two hard filters. Both run before scoring; neither is negotiable.

`affordance_filter` asks whether the body can do the verb at all, and whether
the capability ledger has recorded a recurring failure for it. This is where
negative knowledge does real work: a verb the body keeps failing is not scored
lower, it is never scored.

`viability_filter` asks whether doing it would push any margin under its floor.
It uses `goal.cost_vector`, so a goal that is cheap in aggregate but ruinous on
one axis is still rejected.
"""

from __future__ import annotations

from typing import Any

from body.port import Body
from body.types import Margins
from cortex.goal import Goal
from cortex.ledger import CapabilityLedger

__all__ = ["affordance_filter", "viability_filter"]


def affordance_filter(goal: Goal, ledger: CapabilityLedger, body: Body) -> bool:
    """True when the body declares the verb and it is not a known dead end."""
    verbs = {a.verb for a in body.declare_affordances()}
    if goal.verb not in verbs:
        return False
    if ledger.recurring_failure(goal.verb, {}) is not None:
        return False
    return True


def viability_filter(goal: Goal, margins: Margins) -> bool:
    """False when paying the goal's cost would breach any margin floor."""
    for name, amount in goal.cost_vector.items():
        floor = Margins.FLOORS[name]
        if getattr(margins, name) - float(amount) < floor:
            return False
    return True
