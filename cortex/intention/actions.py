"""From an intention to something a body can actually do.

This is the only place a goal's abstract verb becomes a concrete action dict.
The mapping is deliberately dumb: `OBSERVE(frame, slot)` becomes
`{"verb": "OBSERVE", "slot": slot}`. A verb with no mapping falls back to REST
and logs a warning, because executing nothing is always safe.

The affordance filter is the real gate — a goal whose verb the body does not
declare is removed before it can ever be leased. This fallback exists because
"the filter should have caught it" is not a guarantee, and a body must never
receive an action it cannot parse.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from body.types import Margins, Snapshot

if TYPE_CHECKING:
    from cortex.intention.lease import Intention

__all__ = ["next_action"]

_LOG = logging.getLogger(__name__)

FALLBACK: dict[str, Any] = {"verb": "REST"}


def next_action(
    intention: "Intention", snapshot: Snapshot, margins: Margins
) -> dict[str, Any]:
    """Map an intention's goal onto one body action."""
    del snapshot, margins
    goal = intention.goal
    verb = goal.verb

    if verb == "OBSERVE" and goal.target is not None:
        return {"verb": "OBSERVE", "slot": int(goal.target[1])}
    if verb == "REST":
        return {"verb": "REST"}

    _LOG.warning(
        "no action mapping for goal verb %s (drive %s); falling back to REST",
        verb,
        goal.drive,
    )
    return dict(FALLBACK)
