"""Warm-start tests: a restart is a pause, not an amnesia."""

from __future__ import annotations

from pathlib import Path

from body.embodiments.gridworld import GridWorld
from runtime.boot import boot
from runtime.session import run_session


def test_warm_start_reconstructs_engine_and_ledger(
    tmp_path: Path, low_power_body: GridWorld
) -> None:
    path = tmp_path / "session"
    payloads = run_session(path, low_power_body, 2)
    last = payloads[-1]

    resumed_body = GridWorld.from_state(last["body_state"])
    engine, ledger = boot(path, resumed_body)

    states = {name: state.to_dict() for name, state in engine.drive_states.items()}
    assert states == last["drive_states"]
    assert len(ledger.attempts()) == 2
    assert engine.config.theta_commit == 0.35


def test_partially_completed_intention_is_resumed(
    tmp_path: Path, low_power_body: GridWorld
) -> None:
    path = tmp_path / "session"
    payloads = run_session(path, low_power_body, 2)
    recorded = payloads[-1]["intention"]
    assert recorded is not None
    assert recorded["lease_until"] > 2

    body = GridWorld.from_state(payloads[-1]["body_state"])
    engine, _ = boot(path, body)
    assert engine.current is not None
    assert engine.current.to_dict() == recorded

    more = run_session(path, body, 1)
    assert more[0]["action"] == {"verb": "REST"}
    assert more[0]["intention"]["lease_until"] == recorded["lease_until"]
    assert more[0]["intention_events"] == []


def test_expired_intention_is_not_resumed(
    tmp_path: Path, low_power_body: GridWorld
) -> None:
    path = tmp_path / "session"
    payloads = run_session(path, low_power_body, 3)
    last = payloads[-1]
    assert last["intention"] is not None
    assert last["intention"]["lease_until"] <= last["tick"] + 1

    body = GridWorld.from_state(last["body_state"])
    engine, _ = boot(path, body)
    assert engine.current is None
