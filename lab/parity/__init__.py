"""Parity harness: does the same core run the same way on every body?

Two bodies are only interchangeable if the tensor they hand the core has the
same structure: same width, same slot addresses, same field types, same
residual arity. Values may differ (a rig sees a different room); structure may
not. This module enforces structure, and also enforces that a fresh body
instance reproduces a recorded session's structure tick by tick.
"""

from __future__ import annotations

from typing import Any

from body.types import CELL_FIELDS, Cell, Snapshot
from lab.replay import Replay
from runtime.tick import execute_tick

__all__ = [
    "ParityError",
    "assert_agent_parity",
    "assert_fresh_body_parity",
    "assert_session_parity",
    "assert_structural_parity",
    "structural_signature",
]


class ParityError(AssertionError):
    """Two snapshots do not describe the same tensor shape."""


def _cell_signature(cell: Cell) -> tuple[Any, ...]:
    return (
        cell.frame,
        cell.slot,
        type(cell.value).__name__,
        type(cell.var).__name__,
        type(cell.observed_at).__name__,
        type(cell.freshness).__name__,
        type(cell.modality_mask).__name__,
    )


def structural_signature(snapshot: Snapshot) -> dict[str, Any]:
    """Everything about a snapshot that must match across embodiments."""
    return {
        "body_id": snapshot.body_id,
        "width": snapshot.width,
        "residual_width": len(snapshot.residual),
        "fields": CELL_FIELDS,
        "cells": tuple(_cell_signature(c) for c in snapshot.cells),
        "residual_types": tuple(type(r).__name__ for r in snapshot.residual),
        "residual_present": tuple(r is not None for r in snapshot.residual),
    }


def assert_structural_parity(
    recorded: Snapshot, replayed: Snapshot, *, tick: int | None = None
) -> None:
    """Raise `ParityError` naming the first structural mismatch."""
    left = structural_signature(recorded)
    right = structural_signature(replayed)
    where = f"tick {tick}" if tick is not None else "snapshot"

    if left["width"] != right["width"]:
        raise ParityError(
            f"{where}: tensor width {left['width']} != {right['width']}"
        )
    if left["residual_width"] != right["residual_width"]:
        raise ParityError(
            f"{where}: residual width {left['residual_width']} "
            f"!= {right['residual_width']}"
        )
    if left["cells"] != right["cells"]:
        for index, (a, b) in enumerate(zip(left["cells"], right["cells"])):
            if a != b:
                raise ParityError(
                    f"{where}: slot {index} differs: {a!r} != {b!r}"
                )
        raise ParityError(f"{where}: cell count differs")
    if left["residual_types"] != right["residual_types"]:
        raise ParityError(f"{where}: residual element types differ")
    if not all(left["residual_present"]):
        raise ParityError(f"{where}: residual contains a null entry")


def assert_session_parity(path: Any, tick_range: tuple[int, int] | None = None) -> int:
    """Replay a session and compare structure tick by tick. Returns tick count."""
    replay = Replay(path, tick_range=tick_range)
    recorded = replay.recorded_snapshots()
    replayed = replay.replayed_snapshots()
    if len(recorded) != len(replayed):
        raise ParityError(
            f"tick count {len(recorded)} != {len(replayed)}"
        )
    for index, (a, b) in enumerate(zip(recorded, replayed)):
        assert_structural_parity(a, b, tick=index)
    return len(recorded)


def assert_agent_parity(path: Any) -> int:
    """Replay a full agent session and demand byte-identical tick payloads.

    This is the load-bearing check for the loop: the drives, the arbiter, the
    lease, the ledger, and the body must all reproduce, not just the picture.
    Returns the number of ticks replayed.
    """
    replay = Replay(path)
    if not replay.is_agent:
        raise ParityError("not an agent session: no intention in tick payload")
    replay.verify_agent()
    return len(replay.ticks)


def assert_fresh_body_parity(path: Any) -> int:
    """Run a brand-new body instance over recorded actions; compare structure.

    This is the "same core, no code changes" claim made testable without a rig:
    the body is rebuilt from the session header, not from the replay cache.
    """
    replay = Replay(path)
    header = replay.session.payload
    bodies = replay.bodies
    cls = bodies[str(header.get("body_class", "GridWorld"))]
    body = cls.from_state(header["state"])
    recorded = replay.recorded_snapshots()
    for index, action in enumerate(replay.actions()):
        produced = Snapshot.from_dict(execute_tick(body, action)["snapshot"])
        assert_structural_parity(recorded[index], produced, tick=index)
    return len(recorded)
