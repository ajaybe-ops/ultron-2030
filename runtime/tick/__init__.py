"""The tick kernel: one deterministic step of the agent loop.

Two loops live here.

`execute_tick` is the recording-only loop from the foundation build: act, sense,
package. It is kept because the replay and parity tests for that build are the
regression suite for the substrate, and rewriting them would hide a break.

`agent_tick` is the real loop: sense, arbitrate, act, record. It senses at the
start of the tick, hands the snapshot to the engine, turns the returned
intention into an action, and senses again after the body has moved. The
`body_state` it records is captured *between* `actuate` and the closing
`sense`, which is the state replay needs: the next tick's opening sense is a
cache hit in live execution, and a freshly reconstructed body has no cache, so
it must be handed the pre-sense state and allowed to compute the snapshot
itself.

`intention_events` is derived, not stored on the engine. The engine keeps only
`config`, `drive_states`, and `current`; the tick compares the intention before
and after `step` and reports what changed. That keeps the engine replayable and
still gives the log a readable narrative.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Mapping

from body.port import Body
from body.types import Margins, Snapshot, canonical_bytes
from cortex.arbitration.engine import Engine
from cortex.intention.lease import Intention
from cortex.ledger import CapabilityLedger
from substrate.format.substrate import Record, Substrate

__all__ = [
    "agent_tick",
    "canonical_bytes",
    "digest",
    "execute_tick",
    "record_session",
    "tick_records",
]


def digest(obj: Any) -> str:
    """SHA-256 of the canonical bytes, for provenance and fast comparison."""
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


def execute_tick(body: Body, action: Mapping[str, Any]) -> dict[str, Any]:
    """Actuate, then sense, then package the tick for the log.

    Order matters: margins are spent by `actuate`, and the snapshot is taken
    after the world has moved. A body whose `sense()` is not idempotent at a
    fixed clock reading cannot be replayed and is out of contract.
    """
    result = body.actuate(dict(action))
    after = body.sense()
    state_fn = getattr(body, "state_dict", None)
    payload: dict[str, Any] = {
        "action": dict(action),
        "verb": result.verb,
        "ok": result.ok,
        "detail": result.detail,
        "margin_delta": dict(result.margin_delta),
        "snapshot": after.to_dict(),
        "margins": body.margins().as_dict(),
    }
    if callable(state_fn):
        payload["body_state"] = state_fn()
    return payload


def _event(name: str, intention: Intention, tick: int) -> dict[str, Any]:
    target = intention.goal.target
    return {
        "event": name,
        "tick": int(tick),
        "drive": intention.goal.drive,
        "verb": intention.goal.verb,
        "target": list(target) if target is not None else None,
        "lease_until": int(intention.lease_until),
        "drift": bool(intention.drift),
    }


def _transition_events(
    before: Intention | None,
    after: Intention | None,
    tick: int,
    snapshot: Snapshot,
    margins: Margins,
) -> list[dict[str, Any]]:
    """Describe what happened to the intention during this step."""
    if before is None and after is None:
        return []
    if before is None:
        return [_event("acquired", after, tick)]  # type: ignore[arg-type]
    if before is after:
        return []

    events: list[dict[str, Any]] = []
    if before.check_success(tick, snapshot, margins):
        events.append(_event("completed", before, tick))
    elif after is not None and after.goal != before.goal:
        events.append(_event("preempted", before, tick))
    else:
        events.append(_event("aborted", before, tick))
    if after is not None:
        events.append(_event("acquired", after, tick))
    return events


def agent_tick(
    engine: Engine,
    body: Body,
    ledger: CapabilityLedger,
    tick_index: int,
) -> dict[str, Any]:
    """One full agent step: sense, arbitrate, act, record."""
    snapshot = body.sense()
    margins = body.margins()
    before = engine.current

    intention = engine.step(snapshot, margins, ledger, tick_index, body=body)
    after = engine.current
    events = _transition_events(before, after, tick_index, snapshot, margins)

    if intention is None:
        action: dict[str, Any] = {"verb": "REST"}
    else:
        action = intention.next_action(snapshot, margins)

    result = body.actuate(dict(action))
    ledger.record(
        action["verb"], result.ok, result.detail, tick_index, preconditions={}
    )

    body_state = body.state_dict() if callable(getattr(body, "state_dict", None)) else None
    final = body.sense()

    payload: dict[str, Any] = {
        "tick": int(tick_index),
        "action": dict(action),
        "intention": intention.to_dict() if intention is not None else None,
        "intention_events": events,
        "result": result.to_dict(),
        "snapshot": final.to_dict(),
        "margins": body.margins().as_dict(),
        "drive_states": {
            name: state.to_dict() for name, state in engine.drive_states.items()
        },
    }
    if body_state is not None:
        payload["body_state"] = body_state
    return payload


def record_session(
    substrate: Substrate,
    body: Body,
    actions: Iterable[Mapping[str, Any]],
    meta: Mapping[str, Any] | None = None,
) -> list[Record]:
    """Record a session header and then one tick record per action."""
    header_snapshot = body.sense()
    state_fn = getattr(body, "state_dict", None)
    header: dict[str, Any] = {
        "body_id": header_snapshot.body_id,
        "body_class": type(body).__name__,
        "meta": dict(meta or {}),
    }
    if callable(state_fn):
        header["state"] = state_fn()

    written = [
        substrate.append(
            {
                "t_mono": header_snapshot.t_mono,
                "t_wall": header_snapshot.t_wall,
                "kind": "session",
                "provenance": {"rule": "session.start", "source": "runtime.tick"},
                "payload": header,
            }
        )
    ]

    for action in actions:
        payload = execute_tick(body, action)
        written.append(
            substrate.append(
                {
                    "t_mono": payload["snapshot"]["t_mono"],
                    "t_wall": payload["snapshot"]["t_wall"],
                    "kind": "tick",
                    "provenance": {
                        "rule": "tick",
                        "source": "runtime.tick",
                        "body_id": payload["snapshot"]["body_id"],
                        "snapshot_digest": digest(payload["snapshot"]),
                    },
                    "payload": payload,
                }
            )
        )
    return written


def tick_records(substrate: Substrate, t_from: float | None = None,
                 t_to: float | None = None) -> list[Record]:
    """Tick records in log order, optionally bounded by monotonic time."""
    records = substrate.read_all() if t_from is None else substrate.read_as_of(t_from)
    out = [r for r in records if r.kind == "tick"]
    if t_to is not None:
        out = [r for r in out if r.t_mono <= t_to + 1e-9]
    return out
