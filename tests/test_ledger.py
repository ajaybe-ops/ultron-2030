"""Ledger tests: negative knowledge must suppress, not merely report."""

from __future__ import annotations

from pathlib import Path

from body.embodiments.gridworld import GridWorld
from cortex.arbitration.filters import affordance_filter
from cortex.goal import Goal
from cortex.ledger import CapabilityLedger
from substrate.format.substrate import Substrate


def observe_goal() -> Goal:
    return Goal(
        verb="OBSERVE",
        drive="D1",
        value=1.0,
        cost_vector={"compute": 0.002},
        reversibility=1.0,
        latency=1,
        expected_information_gain=1.0,
        target=("grid", 5),
    )


def test_two_failures_suppress_the_verb() -> None:
    ledger = CapabilityLedger()
    body = GridWorld(seed=1)
    assert affordance_filter(observe_goal(), ledger, body) is True

    ledger.record("OBSERVE", False, "OUT_OF_RANGE", 0, {})
    assert affordance_filter(observe_goal(), ledger, body) is True

    ledger.record("OBSERVE", False, "OUT_OF_RANGE", 1, {})
    failure = ledger.recurring_failure("OBSERVE", {})
    assert failure is not None
    assert failure.signature == "OUT_OF_RANGE"
    assert failure.count == 2
    assert affordance_filter(observe_goal(), ledger, body) is False


def test_success_clears_a_recurring_failure() -> None:
    ledger = CapabilityLedger()
    ledger.record("OBSERVE", False, "OUT_OF_RANGE", 0, {})
    ledger.record("OBSERVE", False, "OUT_OF_RANGE", 1, {})
    assert ledger.can("OBSERVE") is False

    ledger.record("OBSERVE", True, "OBSERVED:5", 2, {})
    assert ledger.can("OBSERVE") is True
    assert ledger.recurring_failure("OBSERVE", {}) is None


def test_different_signatures_do_not_add_up() -> None:
    ledger = CapabilityLedger()
    ledger.record("MOVE", False, "BLOCKED", 0, {})
    ledger.record("MOVE", False, "NO_OP", 1, {})
    assert ledger.can("MOVE") is True
    assert ledger.recurring_failure("MOVE", {}) is None


def test_ledger_round_trips_through_the_substrate(tmp_path: Path) -> None:
    path = tmp_path / "ledger"
    substrate = Substrate(path)
    ledger = CapabilityLedger(substrate=substrate)
    ledger.record("MOVE", False, "BLOCKED", 0, {})
    ledger.record("MOVE", False, "BLOCKED", 1, {})

    rebuilt = CapabilityLedger.from_substrate(substrate)
    failure = rebuilt.recurring_failure("MOVE", {})
    assert failure is not None
    assert (failure.signature, failure.tick, failure.count) == ("BLOCKED", 1, 2)
    assert len(rebuilt.attempts("MOVE")) == 2
