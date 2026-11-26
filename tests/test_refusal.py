"""Refusal: the four answers, the five-part record, and byte-identical replay."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from body.types import Margins
from cortex.goal import Goal
from cortex.intention.lease import intention_for_goal
from cortex.intention.registry import make as make_test
from cortex.selfmodel import SelfModel
from negotiation import (
    Command,
    Decision,
    RefusalMachine,
    RefusalState,
    decide,
    pressures_from_margins,
    replay,
)
from substrate.format.substrate import Substrate


def healthy() -> Margins:
    return Margins(
        power=0.8,
        thermal=0.8,
        compute=0.8,
        storage=0.8,
        sensor_liveness=1.0,
    )


def goal(
    verb: str = "OBSERVE",
    target: tuple[str, int] | None = ("grid", 4),
    drive: str = "D1",
) -> Goal:
    return Goal(
        verb=verb,
        drive=drive,
        value=0.8,
        cost_vector={"power": 0.001, "compute": 0.002},
        reversibility=1.0,
        latency=2,
        expected_information_gain=0.5,
        target=target,
    )


def intention(invariants: tuple = ()):
    return replace(intention_for_goal(goal(), 0), invariants=invariants)


def model_with(*verbs: str) -> SelfModel:
    model = SelfModel()
    for verb in verbs:
        model.record(verb, True, "", 0)
    return model


def snapshot(*verbs: str):
    return model_with(*verbs).snapshot(0)


def test_capability_absent_refuses_with_unmodelable_state() -> None:
    decision = decide(
        intention(),
        Command("GRASP", tick=1),
        SelfModel().snapshot(0),
        pressures_from_margins(healthy()),
    )

    assert decision.outcome is Decision.REFUSE_WITH_REASON
    assert decision.state is RefusalState.UNMODELABLE
    assert decision.record.outranking_drive == "D1"
    assert "no procedure" in decision.record.reason


def test_capability_present_accepts() -> None:
    decision = decide(
        None,
        Command("OBSERVE", params={"slot": 4}, target=("grid", 4), tick=1),
        snapshot("OBSERVE"),
        pressures_from_margins(healthy()),
    )

    assert decision.outcome is Decision.ACCEPT
    assert decision.state is RefusalState.OPEN


def test_margin_below_floor_refuses_any_non_restorative_command() -> None:
    low = Margins(
        power=0.2,
        thermal=0.8,
        compute=0.8,
        storage=0.8,
        sensor_liveness=1.0,
    )
    decision = decide(
        intention(),
        Command("MOVE", cost={"power": 0.01}, tick=1),
        snapshot("MOVE"),
        pressures_from_margins(low),
    )

    assert decision.outcome is Decision.REFUSE_WITH_REASON
    assert decision.state is RefusalState.REFUSING
    assert decision.record.outranking_drive == "D2"
    assert decision.record.evidence["margin"] == "power"
    assert decision.record.evidence["margin_value"] == 0.2


def test_rest_is_accepted_when_a_margin_is_below_floor() -> None:
    low = Margins(
        power=0.2,
        thermal=0.8,
        compute=0.8,
        storage=0.8,
        sensor_liveness=1.0,
    )
    decision = decide(
        intention(),
        Command("REST", tick=1),
        snapshot("REST"),
        pressures_from_margins(low),
    )

    assert decision.outcome is Decision.ACCEPT
    assert decision.state is RefusalState.COMMITTED


def test_no_live_intention_accepts_with_open_state() -> None:
    decision = decide(
        None,
        Command("REST", tick=1),
        snapshot("REST"),
        pressures_from_margins(healthy()),
    )

    assert decision.outcome is Decision.ACCEPT
    assert decision.state is RefusalState.OPEN


def test_concordant_command_is_accepted() -> None:
    decision = decide(
        intention(),
        Command("OBSERVE", params={"slot": 4}, target=("grid", 4), tick=1),
        snapshot("OBSERVE"),
        pressures_from_margins(healthy()),
    )

    assert decision.outcome is Decision.ACCEPT
    assert decision.state is RefusalState.CONCORDANT


def test_cheap_orthogonal_command_runs_concurrently() -> None:
    decision = decide(
        intention(),
        Command("REST", tick=1),
        snapshot("OBSERVE", "REST"),
        pressures_from_margins(healthy()),
    )

    assert decision.outcome is Decision.ACCEPT
    assert decision.state is RefusalState.ORTHOGONAL


def test_expensive_orthogonal_command_is_deferred_with_a_ticket() -> None:
    decision = decide(
        intention(),
        Command("PROBE", cost={"compute": 0.2}, tick=1),
        snapshot("OBSERVE", "PROBE"),
        pressures_from_margins(healthy()),
    )

    assert decision.outcome is Decision.DEFER
    assert decision.state is RefusalState.DEFERRING
    assert decision.record.ticket == "t-0001-probe"


def test_cost_only_conflict_negotiates() -> None:
    decision = decide(
        intention(),
        Command("OBSERVE", params={"slot": 9}, target=("grid", 9), tick=1),
        snapshot("OBSERVE"),
        pressures_from_margins(healthy()),
    )

    assert decision.outcome is Decision.NEGOTIATE
    assert decision.state is RefusalState.NEGOTIATING
    assert "lease ends" in decision.record.alternative


def test_invariant_conflict_refuses() -> None:
    held = intention(
        invariants=(make_test("margin_recovered", name="power", threshold=0.9),)
    )
    decision = decide(
        held,
        Command("OBSERVE", params={"slot": 9}, target=("grid", 9), tick=1),
        snapshot("OBSERVE"),
        pressures_from_margins(healthy()),
    )

    assert decision.outcome is Decision.REFUSE_WITH_REASON
    assert decision.state is RefusalState.REFUSING
    assert decision.record.outranking_drive == "D1"


def test_refusal_record_carries_the_five_required_facts() -> None:
    decision = decide(
        intention(),
        Command("MOVE", cost={"power": 0.01}, tick=1),
        snapshot("MOVE"),
        pressures_from_margins(
            Margins(
                power=0.2,
                thermal=0.8,
                compute=0.8,
                storage=0.8,
                sensor_liveness=1.0,
            )
        ),
    )
    record = decision.record

    assert record.command["verb"] == "MOVE"
    assert record.intention is not None
    assert record.outranking_drive == "D2"
    assert record.reason
    assert record.alternative
    assert record.evidence["tick"] == 1


def test_utterance_has_five_parts() -> None:
    decision = decide(
        intention(),
        Command("MOVE", cost={"power": 0.01}, tick=1),
        snapshot("MOVE"),
        pressures_from_margins(
            Margins(
                power=0.2,
                thermal=0.8,
                compute=0.8,
                storage=0.8,
                sensor_liveness=1.0,
            )
        ),
    )

    assert len(decision.record.utterance().split(" · ")) == 5


def test_repeated_command_without_new_information_is_contested() -> None:
    decision = decide(
        intention(),
        Command(
            "OBSERVE",
            params={"slot": 9},
            target=("grid", 9),
            tick=3,
            repeat=True,
        ),
        snapshot("OBSERVE"),
        pressures_from_margins(healthy()),
    )

    assert decision.state is RefusalState.CONTESTED
    assert decision.outcome is Decision.REFUSE_WITH_REASON
    assert "new information" in decision.record.reason


def test_refusal_replays_byte_identically_from_its_own_record() -> None:
    decision = decide(
        intention(),
        Command("MOVE", cost={"power": 0.01}, tick=1),
        snapshot("MOVE"),
        pressures_from_margins(
            Margins(
                power=0.2,
                thermal=0.8,
                compute=0.8,
                storage=0.8,
                sensor_liveness=1.0,
            )
        ),
    )

    assert replay(decision.record).canonical_bytes() == decision.record.canonical_bytes()


def test_machine_persists_a_replayable_refusal_record(tmp_path: Path) -> None:
    substrate = Substrate(tmp_path / "log")
    machine = RefusalMachine(substrate)
    command = Command("MOVE", cost={"power": 0.01}, tick=7)
    machine.receive(
        command,
        intention(),
        snapshot("MOVE"),
        pressures_from_margins(
            Margins(
                power=0.2,
                thermal=0.8,
                compute=0.8,
                storage=0.8,
                sensor_liveness=1.0,
            )
        ),
    )

    stored = machine.records()[0]
    assert stored.tick == 7
    assert replay(stored).canonical_bytes() == stored.canonical_bytes()
    assert RefusalMachine.from_substrate(substrate).state is RefusalState.REFUSING


def test_decision_does_not_mutate_its_inputs() -> None:
    pressures = pressures_from_margins(healthy())
    snapshot_before = snapshot("OBSERVE").to_dict()
    model = SelfModel()
    model.record("OBSERVE", True, "", 0)
    frozen = model.snapshot(0)

    decide(intention(), Command("REST", tick=1), frozen, pressures)

    assert frozen.to_dict() == snapshot_before
    assert tuple(p.to_dict() for p in pressures) == tuple(
        p.to_dict() for p in pressures_from_margins(healthy())
    )
