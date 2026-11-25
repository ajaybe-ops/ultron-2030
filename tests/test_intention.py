"""Intention tests: leases, success, abort, invariants, serialisation."""

from __future__ import annotations

from body.types import Cell, Margins, Snapshot, modality_bit
from cortex.goal import Goal
from cortex.intention import registry
from cortex.intention.lease import Intention, intention_for_goal

VISION = modality_bit("vision")


def snapshot_with(slot: int = 5, freshness: float = 1.0, var: float = 0.0004) -> Snapshot:
    cells = [
        Cell("grid", index, 0.5, 0.25, 0.0, 0.0, VISION) for index in range(10)
    ]
    cells[slot] = Cell("grid", slot, 0.5, var, 0.0, freshness, VISION)
    return Snapshot(
        t_mono=0.0,
        t_wall=0.0,
        seq=0,
        body_id="test",
        cells=tuple(cells),
        residual=tuple(0.0 for _ in cells),
    )


def healthy() -> Margins:
    return Margins(
        power=1.0, thermal=1.0, compute=1.0, storage=1.0, sensor_liveness=1.0
    )


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


def test_lease_expiry_is_an_abort() -> None:
    intention = intention_for_goal(observe_goal(), 0)
    assert intention.lease_until == 2
    assert intention.check_abort(1, snapshot_with(freshness=0.0), healthy()) is False
    assert intention.check_abort(2, snapshot_with(freshness=0.0), healthy()) is True


def test_success_test_fires_on_a_fresh_target() -> None:
    intention = intention_for_goal(observe_goal(), 0)
    assert intention.check_success(1, snapshot_with(freshness=0.5), healthy()) is False
    assert intention.check_success(1, snapshot_with(freshness=1.0), healthy()) is True


def test_invariant_violation_aborts() -> None:
    intention = Intention(
        goal=observe_goal(),
        lease_until=100,
        invariants=(registry.make("margin_recovered", name="power", threshold=0.99),),
        abort_spec=registry.make("lease_expired", lease_until=100),
    )
    margins = Margins(
        power=0.5, thermal=1.0, compute=1.0, storage=1.0, sensor_liveness=1.0
    )
    assert intention.check_abort(0, snapshot_with(), margins) is True


def test_round_trip_through_the_registry() -> None:
    intention = intention_for_goal(observe_goal(), 7)
    restored = Intention.from_dict(intention.to_dict())
    assert restored == intention
    assert restored.check_success(8, snapshot_with(), healthy()) is True


def test_next_action_maps_observe_to_a_slot() -> None:
    intention = intention_for_goal(observe_goal(), 0)
    assert intention.next_action(snapshot_with(), healthy()) == {
        "verb": "OBSERVE",
        "slot": 5,
    }


def test_unmapped_verb_falls_back_to_rest() -> None:
    goal = Goal(
        verb="PROBE",
        drive="D1",
        value=0.5,
        cost_vector={"compute": 0.004},
        reversibility=0.8,
        latency=2,
        expected_information_gain=0.5,
        target=("grid", 5),
    )
    intention = intention_for_goal(goal, 0)
    assert intention.next_action(snapshot_with(), healthy()) == {"verb": "REST"}
