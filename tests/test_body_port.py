"""The Body port contract and the gridworld's compliance with it."""

from __future__ import annotations

import pytest

from body.embodiments.gridworld import (
    GRID_SLOTS,
    PROXIMITY_SLOT,
    SLOT_COUNT,
    GridWorld,
)
from body.port import Body
from body.types import Cell, Margins, Snapshot


def test_body_is_abstract() -> None:
    with pytest.raises(TypeError):
        Body()  # type: ignore[abstract]


def test_gridworld_declares_only_verbs_it_implements() -> None:
    body = GridWorld()
    verbs = {a.verb for a in body.declare_affordances()}
    assert verbs == {"MOVE", "OBSERVE", "REST"}

    observe = next(a for a in body.declare_affordances() if a.verb == "OBSERVE")
    assert observe.params == ("slot",)
    assert observe.cost["compute"] > 0


def test_margins_decrease_with_movement() -> None:
    body = GridWorld()
    before = body.margins()
    result = body.actuate({"verb": "MOVE", "dx": 1, "dy": 0})
    after = body.margins()

    assert result.ok is True
    assert after.power < before.power
    assert after.thermal < before.thermal
    assert after.sensor_liveness == 1.0


def test_rest_recovers_power_and_thermal() -> None:
    body = GridWorld()
    for _ in range(5):
        body.actuate({"verb": "MOVE", "dx": 1, "dy": 0})
    drained = body.margins()

    body.actuate({"verb": "REST"})
    rested = body.margins()

    assert rested.power > drained.power
    assert rested.thermal > drained.thermal


def test_snapshot_fields_are_valid() -> None:
    body = GridWorld(seed=3)
    snapshot = body.sense()

    assert isinstance(snapshot, Snapshot)
    assert snapshot.body_id == "gridworld-0"
    assert snapshot.seq == 0
    assert snapshot.t_mono == 0.0
    assert snapshot.t_wall == body.t_wall0
    assert snapshot.width == SLOT_COUNT == GRID_SLOTS + 1
    assert len(snapshot.residual) == SLOT_COUNT
    assert all(isinstance(c, Cell) for c in snapshot.cells)
    assert snapshot.cell("proximity", PROXIMITY_SLOT).slot == PROXIMITY_SLOT
    assert all(0.0 <= c.freshness <= 1.0 for c in snapshot.cells)


def test_unknown_verb_fails_without_moving_the_clock() -> None:
    body = GridWorld()
    result = body.actuate({"verb": "FLY", "altitude": 10})

    assert result.ok is False
    assert result.detail == "UNKNOWN_VERB"
    assert body.tick == 0
    assert result.margin_delta == {}


def test_move_is_clamped_at_the_boundary() -> None:
    body = GridWorld()
    result = body.actuate({"verb": "MOVE", "dx": -1, "dy": -1})

    assert result.ok is True
    assert result.detail == "MOVED:0,0"
    assert (body.x, body.y) == (0, 0)


def test_move_out_of_range_is_refused() -> None:
    body = GridWorld()
    result = body.actuate({"verb": "MOVE", "dx": 4, "dy": 0})

    assert result.ok is False
    assert result.detail == "OUT_OF_RANGE"
    assert body.tick == 0


def test_observe_refreshes_a_distant_cell() -> None:
    body = GridWorld()
    far = 99
    body.sense()
    assert body.sense().cells[far].freshness == 0.0

    body.actuate({"verb": "OBSERVE", "slot": far})
    refreshed = body.sense().cells[far]

    assert refreshed.freshness == 1.0
    assert refreshed.var < 0.01
    assert refreshed.observed_at == body.t_mono


def test_margins_stay_inside_the_unit_interval() -> None:
    body = GridWorld()
    for _ in range(40):
        body.actuate({"verb": "MOVE", "dx": 1, "dy": 1})
        body.actuate({"verb": "REST"})
    margins = body.margins()
    assert isinstance(margins, Margins)
    for name, value in margins.as_dict().items():
        assert 0.0 <= value <= 1.0, name
