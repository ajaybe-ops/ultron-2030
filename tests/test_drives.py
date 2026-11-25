"""Drive tests: urgency formulas, VITAL, refractory damping, quiet D3."""

from __future__ import annotations

import pytest

from body.types import Cell, Margins, Snapshot, modality_bit
from cortex.arbitration.config import ArbitrationConfig
from cortex.arbitration.engine import Engine
from cortex.drives import epistemic, homeostatic, legibility
from cortex.drives.state import DriveState
from cortex.ledger import CapabilityLedger

VISION = modality_bit("vision")


def cell(
    slot: int,
    *,
    var: float = 0.25,
    freshness: float = 0.0,
    value: float = 0.5,
    frame: str = "grid",
) -> Cell:
    return Cell(
        frame=frame,
        slot=slot,
        value=value,
        var=var,
        observed_at=0.0,
        freshness=freshness,
        modality_mask=VISION,
    )


def snapshot(cells: list[Cell], residuals: list[float] | None = None) -> Snapshot:
    if residuals is None:
        residuals = [0.0] * len(cells)
    return Snapshot(
        t_mono=0.0,
        t_wall=0.0,
        seq=0,
        body_id="test",
        cells=tuple(cells),
        residual=tuple(residuals),
    )


def healthy() -> Margins:
    return Margins(
        power=1.0, thermal=1.0, compute=1.0, storage=1.0, sensor_liveness=1.0
    )


def test_epistemic_urgency_matches_hand_computation() -> None:
    snap = snapshot([cell(i) for i in range(4)], residuals=[0.1] * 4)
    value = epistemic.urgency(snap, healthy(), CapabilityLedger(), DriveState())
    # r_hat = 1.0 (one sample is its own p95), h_hat = 1.0, c_hat = 0.0
    assert value == pytest.approx(0.5 * 1.0 + 0.3 * 1.0 - 0.2 * 0.0)
    assert value == pytest.approx(0.8)


def test_epistemic_coverage_brake_floors_urgency_at_zero() -> None:
    snap = snapshot([cell(i, var=0.0004, freshness=1.0) for i in range(4)])
    value = epistemic.urgency(snap, healthy(), CapabilityLedger(), DriveState())
    # Nothing is surprising, nothing is frontier, everything is covered.
    assert value == 0.0


def test_epistemic_urgency_rises_with_frontier_variance() -> None:
    state_low = DriveState()
    state_high = DriveState()
    low = snapshot([cell(i, var=0.001, freshness=0.0) for i in range(4)])
    high = snapshot([cell(i, var=0.25, freshness=0.0) for i in range(4)])
    assert epistemic.urgency(high, healthy(), CapabilityLedger(), state_high) > (
        epistemic.urgency(low, healthy(), CapabilityLedger(), state_low)
    )


def test_epistemic_candidates_rank_by_information_per_cost() -> None:
    snap = snapshot(
        [
            cell(0, var=0.25, freshness=0.0),
            cell(1, var=0.02, freshness=0.0),
        ]
    )
    goals = epistemic.candidates(snap, healthy(), CapabilityLedger(), DriveState())
    assert len(goals) == 2
    ratios = [
        goal.expected_information_gain / goal.cost() for goal in goals
    ]
    assert ratios == sorted(ratios, reverse=True)
    assert goals[0].target == ("grid", 0)


def test_epistemic_emits_probe_for_fresh_uncertain_cell() -> None:
    snap = snapshot([cell(0, var=0.05, freshness=1.0)])
    goals = epistemic.candidates(snap, healthy(), CapabilityLedger(), DriveState())
    assert [goal.verb for goal in goals] == ["PROBE"]


def test_homeostatic_urgency_is_the_worst_margin() -> None:
    margins = Margins(
        power=0.3, thermal=1.0, compute=1.0, storage=1.0, sensor_liveness=1.0
    )
    value = homeostatic.urgency(
        snapshot([cell(0)]), margins, CapabilityLedger(), DriveState()
    )
    assert value == pytest.approx(0.7)


def test_homeostatic_vital_latches_on_floor_breach() -> None:
    state = DriveState()
    margins = Margins(
        power=0.2, thermal=1.0, compute=1.0, storage=1.0, sensor_liveness=1.0
    )
    homeostatic.urgency(snapshot([cell(0)]), margins, CapabilityLedger(), state)
    assert state.vital is True
    assert state.deficit == 1.0

    healthy_state = DriveState()
    homeostatic.urgency(snapshot([cell(0)]), healthy(), CapabilityLedger(), healthy_state)
    assert healthy_state.vital is False


def test_homeostatic_emits_every_viability_verb() -> None:
    goals = homeostatic.candidates(
        snapshot([cell(0)]), healthy(), CapabilityLedger(), DriveState()
    )
    assert {goal.verb for goal in goals} == {
        "REST",
        "SHED_LOAD",
        "SEEK_CHARGE",
        "COOL",
        "REACQUIRE_SENSOR",
    }


def test_legibility_is_quiet_when_no_presence_is_sensed() -> None:
    state = DriveState()
    snap = snapshot([cell(0), cell(1)])
    assert legibility.presence(snap) == 0.0
    assert legibility.urgency(snap, healthy(), CapabilityLedger(), state) == 0.0
    goals = legibility.candidates(snap, healthy(), CapabilityLedger(), state)
    assert all(goal.verb in {"DISCLOSE", "APPROACH", "NEGOTIATE"} for goal in goals)


def test_legibility_reads_presence_slots() -> None:
    snap = snapshot([cell(0, frame="presence", freshness=0.8)])
    assert legibility.presence(snap) == pytest.approx(0.8)
    value = legibility.urgency(snap, healthy(), CapabilityLedger(), DriveState())
    assert value == pytest.approx(0.4 * 0.8)


def test_refractory_window_halves_urgency() -> None:
    snap = snapshot([cell(i) for i in range(4)], residuals=[0.1] * 4)
    margins = Margins(
        power=0.3, thermal=1.0, compute=1.0, storage=1.0, sensor_liveness=1.0
    )

    warm = Engine(ArbitrationConfig())
    damped = Engine(ArbitrationConfig())
    damped.drive_states["D2"].refractory_until = 10

    raw = warm._urgencies(snap, margins, CapabilityLedger(), 0)["D2"]
    half = damped._urgencies(snap, margins, CapabilityLedger(), 0)["D2"]
    assert raw == pytest.approx(0.7)
    assert half == pytest.approx(0.35)
