"""Shared fixtures: a recorded session, an agent session, and a tamper helper."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from body.embodiments.gridworld import GridWorld
from runtime.session import run_session
from runtime.tick import record_session
from substrate.format.substrate import Substrate

ACTIONS: list[dict] = (
    [{"verb": "MOVE", "dx": 1, "dy": 0}] * 4
    + [{"verb": "OBSERVE", "slot": 42}]
    + [{"verb": "MOVE", "dx": 0, "dy": 1}] * 4
    + [{"verb": "REST"}]
    + [{"verb": "MOVE", "dx": 1, "dy": 1}]
    + [{"verb": "MOVE", "dx": -1, "dy": 0}]
    + [{"verb": "OBSERVE", "slot": 99}]
    + [{"verb": "REST"}]
    + [{"verb": "MOVE", "dx": 0, "dy": -1}] * 2
    + [{"verb": "MOVE", "dx": -1, "dy": -1}] * 2
    + [{"verb": "MOVE", "dx": 0, "dy": 1}]
    + [{"verb": "REST"}]
)

assert len(ACTIONS) == 20


def low_power_state(seed: int = 11) -> dict[str, Any]:
    """A gridworld state whose power margin starts under its floor.

    VITAL has to fire on tick 0 for the loop test to prove that survival
    outranks curiosity, so the body is rebuilt from a state rather than
    drained over several ticks.
    """
    body = GridWorld(seed=seed)
    state = body.state_dict()
    state["margins"]["power"] = 0.2
    return state


@pytest.fixture
def sub_path(tmp_path: Path) -> Path:
    return tmp_path / "substrate"


@pytest.fixture
def session(tmp_path: Path) -> dict:
    path = tmp_path / "session"
    substrate = Substrate(path)
    body = GridWorld(seed=11)
    records = record_session(substrate, body, ACTIONS, meta={"purpose": "test"})
    return {
        "path": path,
        "substrate": substrate,
        "body": body,
        "records": records,
        "actions": list(ACTIONS),
    }


@pytest.fixture
def low_power_body() -> GridWorld:
    """A fresh body that begins one tick away from a power-floor breach."""
    return GridWorld.from_state(low_power_state())


@pytest.fixture(scope="session")
def agent_session(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """A 200-tick agent session with no user input at all.

    Session-scoped: the loop and its replay are the two most expensive checks
    in the suite, and they must run against the same recording.
    """
    path = tmp_path_factory.mktemp("agent-session") / "substrate"
    body = GridWorld.from_state(low_power_state())
    payloads = run_session(path, body, 200)
    return {"path": path, "payloads": payloads, "body": body}


def tamper_tick_snapshot(path: Path, tick_index: int, slot: int,
                         value: float = 0.1234567) -> None:
    """Rewrite one recorded cell value so replay must notice the lie."""
    shard = sorted((Path(path) / "shards").glob("shard-*.jsonl"))[-1]
    lines = [line for line in shard.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [json.loads(line) for line in lines]
    ticks = [r for r in records if r["kind"] == "tick"]
    ticks[tick_index]["payload"]["snapshot"]["cells"][slot]["value"] = value
    shard.write_text(
        "\n".join(
            json.dumps(r, sort_keys=True, separators=(",", ":")) for r in records
        )
        + "\n",
        encoding="utf-8",
    )
