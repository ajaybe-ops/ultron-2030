"""Cross-layer parity: fusion, refusal, and self-model must not depend on a body."""

from __future__ import annotations

import pytest

from body.embodiments.gridworld import GridWorld
from body.types import Margins
from cortex.goal import Goal
from cortex.intention.lease import intention_for_goal
from cortex.selfmodel import SelfModel
from lab.parity import (
    ParityError,
    assert_fusion_parity,
    assert_fusion_shape_parity,
    assert_layered_parity,
    assert_refusal_parity,
    assert_selfmodel_parity,
    trace_layers,
)
from negotiation import Command, pressures_from_margins
from sensorium.fusion import (
    GridWorldBundle,
    PiPerceptionBundle,
    Reading,
    StaticBundle,
    fuse,
)


def healthy() -> Margins:
    return Margins(
        power=0.8,
        thermal=0.8,
        compute=0.8,
        storage=0.8,
        sensor_liveness=1.0,
    )


def intention():
    goal = Goal(
        verb="OBSERVE",
        drive="D1",
        value=0.8,
        cost_vector={"power": 0.001, "compute": 0.002},
        reversibility=1.0,
        latency=2,
        expected_information_gain=0.5,
        target=("grid", 4),
    )
    return intention_for_goal(goal, 0)


def snapshot():
    model = SelfModel()
    for verb in ("OBSERVE", "MOVE", "REST"):
        model.record(verb, True, "", 0)
    return model.snapshot(1)


def commands() -> list[Command]:
    return [
        Command("OBSERVE", params={"slot": 4}, target=("grid", 4), tick=1),
        Command("OBSERVE", params={"slot": 9}, target=("grid", 9), tick=2),
        Command("MOVE", cost={"power": 0.01}, tick=3),
    ]


def bundle(t: float = 1.0) -> StaticBundle:
    return StaticBundle(
        body_id="body-0",
        t_mono=t,
        t_wall=t,
        seq=int(t),
        data={
            "occupancy": Reading(0.4, 1.0, t),
            "surprise": Reading(0.1, 1.0, t),
            "viability": Reading(0.8, 1.0, t),
            "presence": Reading(1.0, 1.0, t),
            "audio": Reading(0.25, 1.0, t),
        },
    )


def test_refusal_decisions_are_byte_identical_across_bodies() -> None:
    left = trace_layers(
        [bundle()], commands(), intention(), snapshot(), pressures_from_margins(healthy())
    )
    right = trace_layers(
        [bundle()], commands(), intention(), snapshot(), pressures_from_margins(healthy())
    )

    assert assert_refusal_parity(left.refusals, right.refusals) == 3


def test_fusion_of_the_same_bundle_is_byte_identical() -> None:
    left = trace_layers(
        [bundle(1.0), bundle(2.0)],
        [],
        intention(),
        snapshot(),
        pressures_from_margins(healthy()),
    )
    right = trace_layers(
        [bundle(1.0), bundle(2.0)],
        [],
        intention(),
        snapshot(),
        pressures_from_margins(healthy()),
    )

    for a, b in zip(left.fusion, right.fusion):
        assert_fusion_parity(a, b)


def test_gridworld_and_pi_fusion_share_a_shape() -> None:
    body = GridWorld(seed=11)
    grid = fuse(GridWorldBundle.from_snapshot(body.sense(), body.margins()))
    pi = fuse(
        PiPerceptionBundle.from_frames(
            camera=[0.0, 1.0, 0.0, 1.0],
            pir=0.0,
            mic_energy=0.1,
            margins=body.margins(),
            residual=body.sense().residual,
        )
    )

    assert_fusion_shape_parity(grid, pi)


def test_selfmodel_states_are_byte_identical_after_same_attempts() -> None:
    left = SelfModel()
    right = SelfModel()
    for model in (left, right):
        model.record("OBSERVE", True, "", 0)
        model.record("GRASP", False, "slip-after-contact", 1)

    assert_selfmodel_parity(left.snapshot(2), right.snapshot(2))


def test_layered_parity_passes_for_identical_inputs() -> None:
    left = trace_layers(
        [bundle()], commands(), intention(), snapshot(), pressures_from_margins(healthy())
    )
    right = trace_layers(
        [bundle()], commands(), intention(), snapshot(), pressures_from_margins(healthy())
    )

    assert_layered_parity(left, right)


def test_layered_parity_fails_loudly_on_a_different_decision() -> None:
    left = trace_layers(
        [bundle()], commands(), intention(), snapshot(), pressures_from_margins(healthy())
    )
    other = [
        Command("OBSERVE", params={"slot": 4}, target=("grid", 4), tick=99),
    ]
    right = trace_layers(
        [bundle()], other, intention(), snapshot(), pressures_from_margins(healthy())
    )

    with pytest.raises(ParityError, match="refusal"):
        assert_layered_parity(left, right)
