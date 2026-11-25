"""Embodiment layer: the Body port and the data types it speaks."""

from body.port import Body
from body.types import (
    Affordance,
    ActionResult,
    Cell,
    Margins,
    Snapshot,
    canonical_bytes,
)

__all__ = [
    "Affordance",
    "ActionResult",
    "Body",
    "Cell",
    "Margins",
    "Snapshot",
    "canonical_bytes",
]
