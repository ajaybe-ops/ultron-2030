"""Warm start: rebuild the decision state from SUBSTRATE-1.

Booting is not "start fresh and pretend nothing happened". The engine comes
back with the drive states it had, the ledger it learned, and any intention
whose lease has not run out. That is what makes a restart a pause rather than
an amnesia.

Nothing here re-derives state from the body. The body is the caller's problem;
this module reads only the log.
"""

from __future__ import annotations

from typing import Any

from body.port import Body
from cortex.arbitration.config import ArbitrationConfig
from cortex.arbitration.engine import Engine
from cortex.drives.state import DriveState
from cortex.intention.lease import Intention
from cortex.ledger import CapabilityLedger
from substrate.format.substrate import Record, Substrate

__all__ = ["boot", "boot_from", "reconstruct"]


def _session_record(records: list[Record]) -> Record | None:
    return next((record for record in records if record.kind == "session"), None)


def reconstruct(
    substrate: Substrate,
    config: ArbitrationConfig | None = None,
) -> Engine:
    """Rebuild an `Engine` from the records already in `substrate`."""
    records = substrate.read_all()
    session = _session_record(records)

    chosen = config
    if chosen is None and session is not None:
        stored = session.payload.get("config")
        if isinstance(stored, dict) and stored:
            chosen = ArbitrationConfig.from_dict(stored)

    engine = Engine(chosen)

    for record in records:
        if record.kind != "drive":
            continue
        name = str(record.payload.get("drive", ""))
        state = record.payload.get("state")
        if name in engine.drive_states and isinstance(state, dict):
            engine.drive_states[name] = DriveState.from_dict(state)

    ticks = [record for record in records if record.kind == "tick"]
    if ticks:
        payload: dict[str, Any] = ticks[-1].payload
        next_tick = int(payload.get("tick", 0)) + 1
        raw = payload.get("intention")
        if raw:
            intention = Intention.from_dict(raw)
            if intention.lease_until > next_tick:
                engine.current = intention

    return engine


def boot_from(
    substrate: Substrate,
    body: Body,
    config: ArbitrationConfig | None = None,
) -> tuple[Engine, CapabilityLedger]:
    """Rebuild against an already-open substrate.

    A session must not open the log twice: two `Substrate` instances each
    compute their own `_next_seq`, and interleaved appends would reuse `seq`
    values. Everything that writes goes through one handle.
    """
    del body
    ledger = CapabilityLedger.from_substrate(substrate)
    engine = reconstruct(substrate, config)
    return engine, ledger


def boot(
    substrate_path: Any,
    body: Body,
    config: ArbitrationConfig | None = None,
) -> tuple[Engine, CapabilityLedger]:
    """Open the log, rebuild the ledger and engine, and return them ready.

    `body` is accepted so the signature matches the dossier's warm start and so
    a caller cannot boot without one; the engine never reads it here.
    """
    return boot_from(Substrate(substrate_path), body, config)
