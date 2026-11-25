"""Deterministic replay: re-execute a recorded session and prove it matches.

Replay is the project's ground truth. If a tick cannot be re-executed from the
substrate and produce byte-identical output, then the log is not a record of
what happened — it is a story. Every later layer (drives, arbitration,
refusal) is allowed to be non-deterministic only in ways this harness can
name.

Two session shapes are supported:

* **recording-only** sessions, written by `runtime.tick.record_session`, where
  a tick is one action and the compared payload is the snapshot. `verify()`
  and the snapshot accessors serve these, and they are kept so the foundation
  build's tests remain a regression suite for the substrate.
* **agent** sessions, written by `runtime.session.run_session`, where a tick is
  a full arbitration round. `verify_agent()` compares the *entire* tick
  payload, because a loop that produces the same picture but a different
  intention has not been replayed.

The body is reconstructed from the `body_state` the previous tick recorded, so
no external state is consulted. The engine is reconstructed from the previous
tick's drive states and intention; the ledger is rebuilt from `ledger` records
that predate the tick. Replay never writes.
"""

from __future__ import annotations

from typing import Any, Iterator, Mapping, Sequence

from body.embodiments.gridworld import GridWorld
from body.types import Snapshot
from cortex.arbitration.config import ArbitrationConfig
from cortex.arbitration.engine import Engine
from cortex.drives.state import DriveState
from cortex.intention.lease import Intention
from cortex.ledger import CapabilityLedger
from substrate.format.substrate import Record, Substrate
from runtime.tick import agent_tick, canonical_bytes, execute_tick

__all__ = ["Replay", "ReplayDivergence", "ReplayError"]

DEFAULT_BODIES: dict[str, type] = {"GridWorld": GridWorld}


class ReplayError(RuntimeError):
    """The session cannot be replayed at all (missing header, empty range)."""


class ReplayDivergence(AssertionError):
    """A tick re-executed to something other than what was recorded."""

    def __init__(self, tick_index: int, field: str, expected: Any, actual: Any) -> None:
        self.tick_index = int(tick_index)
        self.field = field
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"tick {tick_index} diverged at {field or '<root>'}: "
            f"expected {expected!r}, got {actual!r}"
        )


def _diff_path(expected: Any, actual: Any, prefix: str = "") -> tuple[str, Any, Any]:
    """Locate the first structural difference and describe its path."""
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        keys = list(expected.keys())
        for key in keys:
            if key not in actual:
                return f"{prefix}.{key}" if prefix else str(key), expected[key], None
            path, exp, act = _diff_path(expected[key], actual[key], f"{prefix}.{key}" if prefix else str(key))
            if path:
                return path, exp, act
        for key in actual:
            if key not in expected:
                return f"{prefix}.{key}" if prefix else str(key), None, actual[key]
        return "", expected, actual
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(expected) != len(actual):
            return f"{prefix}.length", len(expected), len(actual)
        for index, (exp_item, act_item) in enumerate(zip(expected, actual)):
            path, exp, act = _diff_path(exp_item, act_item, f"{prefix}[{index}]")
            if path:
                return path, exp, act
        return "", expected, actual
    if expected != actual:
        return prefix, expected, actual
    return "", expected, actual


class Replay:
    """Re-execute a recorded session from SUBSTRATE-1 and compare bytes."""

    def __init__(
        self,
        path: Any,
        tick_range: tuple[int, int] | None = None,
        *,
        shard_size: int = 1000,
        bodies: Mapping[str, type] | None = None,
    ) -> None:
        self.substrate = Substrate(path, shard_size=shard_size)
        self.tick_range = tick_range
        self.bodies = dict(DEFAULT_BODIES)
        if bodies:
            self.bodies.update(bodies)
        self._session, self._ticks = self._load()
        self._bounds()  # fail fast on an impossible window

    # ---------------------------------------------------------------- loading

    @property
    def session(self) -> Record:
        """The session header record."""
        return self._session

    @property
    def ticks(self) -> Sequence[Record]:
        """Every tick record, in log order."""
        return tuple(self._ticks)

    @property
    def is_agent(self) -> bool:
        """True when the tick payload came from `runtime.session`."""
        return "intention" in self._ticks[0].payload

    def _load(self) -> tuple[Record, list[Record]]:
        records = self.substrate.read_all()
        session = next((r for r in records if r.kind == "session"), None)
        if session is None:
            raise ReplayError("no session record in substrate")
        ticks = [r for r in records if r.kind == "tick"]
        if not ticks:
            raise ReplayError("session has no tick records")
        return session, ticks

    def _bounds(self) -> tuple[int, int]:
        last = len(self._ticks) - 1
        if self.tick_range is None:
            return 0, last
        lo, hi = self.tick_range
        if lo < 0 or hi > last or lo > hi:
            raise ReplayError(
                f"tick_range {(lo, hi)} outside 0..{last}"
            )
        return int(lo), int(hi)

    def _body_for(self, state: Mapping[str, Any], class_name: str):
        cls = self.bodies.get(class_name)
        if cls is None:
            raise ReplayError(f"unknown body class {class_name!r}")
        return cls.from_state(state)

    # ------------------------------------------------------------------ replay

    def run(self) -> bytes:
        """Re-execute the range and return the canonical stream."""
        if self.is_agent:
            return self._run_agent()
        return self._run_legacy()

    def _run_legacy(self) -> bytes:
        lo, hi = self._bounds()
        header = self._session.payload
        class_name = str(header.get("body_class", "GridWorld"))
        state = header["state"] if lo == 0 else self._ticks[lo - 1].payload["body_state"]

        out: list[bytes] = []
        for index in range(lo, hi + 1):
            record = self._ticks[index]
            body = self._body_for(state, class_name)
            produced = execute_tick(body, record.payload["action"])

            expected = record.payload["snapshot"]
            actual = produced["snapshot"]
            if canonical_bytes(expected) != canonical_bytes(actual):
                field, exp, act = _diff_path(expected, actual)
                raise ReplayDivergence(index, field, exp, act)

            for name in ("margins", "body_state"):
                if name not in record.payload:
                    continue
                if name not in produced:
                    raise ReplayDivergence(index, name, record.payload[name], None)
                field, exp, act = _diff_path(record.payload[name], produced[name])
                if field:
                    raise ReplayDivergence(index, f"{name}.{field}", exp, act)

            state = produced["body_state"]
            out.append(canonical_bytes(actual))
        return b"\n".join(out)

    def _agent_initial_state(self, lo: int) -> tuple[dict[str, Any], dict[str, DriveState], Intention | None, CapabilityLedger]:
        header = self._session.payload
        if lo == 0:
            body_state = dict(header["state"])
            drive_states = {name: DriveState() for name in ("D1", "D2", "D3")}
            current = None
            ledger = CapabilityLedger()
        else:
            previous = self._ticks[lo - 1].payload
            body_state = dict(previous["body_state"])
            drive_states = {
                name: DriveState.from_dict(state)
                for name, state in previous["drive_states"].items()
            }
            raw = previous.get("intention")
            current = Intention.from_dict(raw) if raw else None
            payloads = [
                record.payload
                for record in self.substrate.read_all()
                if record.kind == "ledger" and int(record.payload.get("tick", -1)) < lo
            ]
            ledger = CapabilityLedger.from_records(payloads)
        return body_state, drive_states, current, ledger

    def _iter_agent(self) -> Iterator[tuple[int, dict[str, Any], dict[str, Any]]]:
        """Yield `(index, recorded_payload, produced_payload)` for the range."""
        lo, hi = self._bounds()
        header = self._session.payload
        class_name = str(header.get("body_class", "GridWorld"))
        stored = header.get("config")
        config = ArbitrationConfig.from_dict(stored) if stored else ArbitrationConfig()

        body_state, drive_states, current, ledger = self._agent_initial_state(lo)

        engine = Engine(config)
        engine.drive_states = drive_states
        engine.current = current

        for index in range(lo, hi + 1):
            body = self._body_for(body_state, class_name)
            produced = agent_tick(engine, body, ledger, index)
            yield index, self._ticks[index].payload, produced
            body_state = produced["body_state"]

    def _run_agent(self) -> bytes:
        out: list[bytes] = []
        for index, expected, produced in self._iter_agent():
            if canonical_bytes(expected) != canonical_bytes(produced):
                field, exp, act = _diff_path(expected, produced)
                raise ReplayDivergence(index, field, exp, act)
            out.append(canonical_bytes(produced))
        return b"\n".join(out)

    def recorded_bytes(self) -> bytes:
        """The canonical stream exactly as it was recorded."""
        lo, hi = self._bounds()
        if self.is_agent:
            return b"\n".join(
                canonical_bytes(self._ticks[i].payload) for i in range(lo, hi + 1)
            )
        return b"\n".join(
            canonical_bytes(self._ticks[i].payload["snapshot"])
            for i in range(lo, hi + 1)
        )

    def verify(self) -> None:
        """Raise unless a recording-only session replays byte for byte."""
        if self.is_agent:
            raise ReplayError("agent session: use verify_agent()")
        if self.run() != self.recorded_bytes():
            raise ReplayError("replay produced different bytes without a field diff")

    def verify_agent(self) -> None:
        """Raise unless an agent session replays byte for byte, whole payload."""
        if not self.is_agent:
            raise ReplayError("recording-only session: use verify()")
        if self.run() != self.recorded_bytes():
            raise ReplayError("replay produced different bytes without a field diff")

    # ------------------------------------------------------------------- reads

    def recorded_snapshots(self) -> list[Snapshot]:
        lo, hi = self._bounds()
        return [
            Snapshot.from_dict(self._ticks[i].payload["snapshot"])
            for i in range(lo, hi + 1)
        ]

    def replayed_snapshots(self) -> list[Snapshot]:
        lo, hi = self._bounds()
        if self.is_agent:
            return [
                Snapshot.from_dict(produced["snapshot"])
                for _, _, produced in self._iter_agent()
            ]
        header = self._session.payload
        class_name = str(header.get("body_class", "GridWorld"))
        state = header["state"] if lo == 0 else self._ticks[lo - 1].payload["body_state"]
        out: list[Snapshot] = []
        for index in range(lo, hi + 1):
            body = self._body_for(state, class_name)
            produced = execute_tick(body, self._ticks[index].payload["action"])
            out.append(Snapshot.from_dict(produced["snapshot"]))
            state = produced["body_state"]
        return out

    def actions(self) -> list[dict[str, Any]]:
        lo, hi = self._bounds()
        return [dict(self._ticks[i].payload["action"]) for i in range(lo, hi + 1)]
