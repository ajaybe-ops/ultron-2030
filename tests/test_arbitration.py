"""Arbitration tests: scoring, thresholds, drift, carry bonus, VITAL."""

from __future__ import annotations

import pytest

from body.embodiments.gridworld import GridWorld
from body.types import Margins
from cortex.arbitration.config import ArbitrationConfig
from cortex.arbitration.engine import Engine
from cortex.arbitration.score import anti_starvation, score
from cortex.drives.state import DriveState
from cortex.goal import Goal
from cortex.intention.lease import intention_for_goal
from cortex.ledger import CapabilityLedger
from runtime.tick import agent_tick


def goal(
    verb: str = "OBSERVE",
    drive: str = "D1",
    *,
    value: float = 1.0,
    cost: dict[str, float] | None = None,
    gain: float = 0.0,
    target: tuple[str, int] | None = ("grid", 0),
) -> Goal:
    return Goal(
        verb=verb,
        drive=drive,
        value=value,
        cost_vector=cost if cost is not None else {"compute": 0.002},
        reversibility=1.0,
        latency=1,
        expected_information_gain=gain,
        target=target,
    )


def healthy() -> Margins:
    return Margins(
        power=1.0, thermal=1.0, compute=1.0, storage=1.0, sensor_liveness=1.0
    )


def test_score_matches_hand_computation() -> None:
    config = ArbitrationConfig()
    state = DriveState(last_served_tick=0)
    candidate = goal(
        value=0.5, cost={"power": 0.1, "thermal": 0.1}, gain=0.4
    )
    # mu = 1.0, urgency = 0.6, value = 0.5, IG factor = 1.2, cost = 0.2
    expected = 1.0 * 0.6 * 0.5 * 1.2 * 1.0 / 1.2
    assert score(candidate, 0.6, state, 0, config) == pytest.approx(expected)
    assert expected == pytest.approx(0.3)


def test_anti_starvation_grows_then_caps() -> None:
    config = ArbitrationConfig()
    state = DriveState(last_served_tick=0)
    assert anti_starvation(state, 0, config) == pytest.approx(1.0)
    assert anti_starvation(state, 40, config) == pytest.approx(1.2)
    assert anti_starvation(state, 10_000, config) == pytest.approx(3.0)


def test_commit_threshold_decides_commit_versus_drift() -> None:
    body = GridWorld(seed=1)
    snap = body.sense()
    ledger = CapabilityLedger()

    eager = Engine(ArbitrationConfig(theta_commit=0.0))
    committed = eager.step(snap, body.margins(), ledger, 0, body=body)
    assert committed is not None
    assert committed.drift is False

    picky = Engine(ArbitrationConfig(theta_commit=100.0))
    drifted = picky.step(snap, body.margins(), ledger, 0, body=body)
    assert drifted is not None
    assert drifted.drift is True
    assert drifted.lease_until == 1


def test_preempt_needs_the_multiplier() -> None:
    config = ArbitrationConfig()
    engine = Engine(config)
    current = intention_for_goal(goal(target=("grid", 99)), 0, lease_ticks=1)
    urgencies = {"D1": 0.5, "D2": 0.5, "D3": 0.0}
    incumbent = score(
        current.goal, urgencies["D1"], engine.drive_states["D1"], 1, config
    )
    challenger = goal(verb="REST", drive="D2", target=None)

    assert (
        engine._should_preempt(
            current, challenger, 1.2 * incumbent, urgencies, 1, healthy()
        )
        is False
    )
    assert (
        engine._should_preempt(
            current, challenger, 1.5 * incumbent, urgencies, 1, healthy()
        )
        is True
    )


def test_asymmetric_thresholds_prevent_thrashing() -> None:
    body = GridWorld(seed=3)
    engine = Engine()
    ledger = CapabilityLedger()
    preemptions = 0
    for tick in range(50):
        payload = agent_tick(engine, body, ledger, tick)
        preemptions += sum(
            1 for event in payload["intention_events"] if event["event"] == "preempted"
        )
    assert preemptions <= 5
    assert ArbitrationConfig().theta_preempt > 1.0


def test_carry_bonus_shortens_refractory_on_abort() -> None:
    engine = Engine()
    intention = intention_for_goal(
        goal(verb="REST", drive="D2", target=None), 0
    )
    engine._release(intention, 10, completed=False)
    assert engine.drive_states["D2"].refractory_until == 10 + 36

    engine._release(intention, 20, completed=True)
    assert engine.drive_states["D2"].refractory_until == 60
    assert engine.drive_states["D2"].served_count == 2


def test_vital_preempts_a_live_epistemic_intention() -> None:
    body = GridWorld(seed=1)
    snap = body.sense()
    engine = Engine()
    # Slot 99 is outside the sensing radius, so this intention cannot have
    # succeeded yet; the only reason it should end is VITAL.
    engine.current = intention_for_goal(goal(target=("grid", 99)), 0, lease_ticks=100)
    margins = Margins(
        power=0.2, thermal=1.0, compute=1.0, storage=1.0, sensor_liveness=1.0
    )

    result = engine.step(snap, margins, CapabilityLedger(), 1, body=body)
    assert result is not None
    assert result.goal.drive == "D2"
    assert result.goal.verb == "REST"


def test_tie_break_is_deterministic_by_target() -> None:
    engine = Engine()
    left = goal(target=("grid", 5))
    right = goal(target=("grid", 2))
    ranked = engine._rank([left, right], {"D1": 1.0}, 0)
    assert [entry[1].target for entry in ranked] == [("grid", 2), ("grid", 5)]


def test_vital_restricts_the_field_to_homeostatic_candidates() -> None:
    body = GridWorld(seed=1)
    snap = body.sense()
    engine = Engine()
    margins = Margins(
        power=0.2, thermal=1.0, compute=1.0, storage=1.0, sensor_liveness=1.0
    )
    engine._urgencies(snap, margins, CapabilityLedger(), 0)
    candidates = engine._candidates(snap, margins, CapabilityLedger(), body)
    assert candidates
    assert {candidate.drive for candidate in candidates} == {"D2"}
