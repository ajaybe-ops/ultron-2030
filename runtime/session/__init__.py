"""Sessions: run the agent for N ticks with nobody in the room.

A session owns the persistence cadence. After every tick it appends one `tick`
record, one `drive` record per drive, and one `intention` record per intention
event. The tick record is the replay contract; the drive and intention records
are what make warm start cheap and the log readable.

The session header is written once and carries the arbitration config, because
a config change mid-session would make replay depend on when the reader asked.
A substrate is assumed to be single-config for its lifetime.
"""

from __future__ import annotations

from typing import Any

from body.port import Body
from cortex.arbitration.config import ArbitrationConfig
from cortex.selfmodel import SelfModel
from runtime.boot import boot_from
from runtime.tick import agent_tick, digest
from substrate.format.substrate import Substrate

__all__ = ["run_session"]


def _write_header(substrate: Substrate, body: Body, config: ArbitrationConfig) -> None:
    """Write the session header, capturing body state *before* the first sense.

    The state is taken first because the first agent tick opens with a
    `sense()`, and a body rebuilt from this header must be able to produce that
    same snapshot. Capturing the state after sensing would record fusion output
    and replay would apply the decay twice.
    """
    state_fn = getattr(body, "state_dict", None)
    state = state_fn() if callable(state_fn) else None
    snapshot = body.sense()
    header: dict[str, Any] = {
        "body_id": snapshot.body_id,
        "body_class": type(body).__name__,
        "config": config.to_dict(),
        "meta": {},
    }
    if state is not None:
        header["state"] = state
    substrate.append(
        {
            "t_mono": snapshot.t_mono,
            "t_wall": snapshot.t_wall,
            "kind": "session",
            "provenance": {"rule": "session.start", "source": "runtime.session"},
            "payload": header,
        }
    )


def _persist(substrate: Substrate, payload: dict[str, Any]) -> None:
    snapshot = payload["snapshot"]
    t_mono = float(snapshot["t_mono"])
    t_wall = float(snapshot["t_wall"])

    substrate.append(
        {
            "t_mono": t_mono,
            "t_wall": t_wall,
            "kind": "tick",
            "provenance": {
                "rule": "agent_tick",
                "source": "runtime.session",
                "body_id": snapshot["body_id"],
                "snapshot_digest": digest(snapshot),
            },
            "payload": payload,
        }
    )

    for name, state in payload["drive_states"].items():
        substrate.append(
            {
                "t_mono": t_mono,
                "t_wall": t_wall,
                "kind": "drive",
                "provenance": {
                    "rule": "drive.state",
                    "source": "runtime.session",
                    "drive": name,
                },
                "payload": {
                    "drive": name,
                    "tick": int(payload["tick"]),
                    "state": state,
                },
            }
        )

    for event in payload["intention_events"]:
        substrate.append(
            {
                "t_mono": t_mono,
                "t_wall": t_wall,
                "kind": "intention",
                "provenance": {
                    "rule": f"intention.{event['event']}",
                    "source": "runtime.session",
                },
                "payload": event,
            }
        )


def run_session(
    substrate_path: Any,
    body: Body,
    max_ticks: int,
    config: ArbitrationConfig | None = None,
) -> list[dict[str, Any]]:
    """Boot, then run `max_ticks` agent ticks, persisting after each one.

    The tick index is taken from `body.tick`, so a body reconstructed from a
    recorded state resumes at the right number instead of restarting at zero.
    """
    if int(max_ticks) < 0:
        raise ValueError("max_ticks must be >= 0")

    chosen = config if config is not None else ArbitrationConfig()
    substrate = Substrate(substrate_path)
    if not any(record.kind == "session" for record in substrate.read_all()):
        _write_header(substrate, body, chosen)

    engine, ledger = boot_from(substrate, body, chosen)
    selfmodel = SelfModel.from_substrate(substrate)

    payloads: list[dict[str, Any]] = []
    for _ in range(int(max_ticks)):
        payload = agent_tick(
            engine, body, ledger, int(body.tick), selfmodel=selfmodel
        )
        _persist(substrate, payload)
        payloads.append(payload)
    return payloads
