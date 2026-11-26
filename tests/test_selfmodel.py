"""Self-model: absence by default, earned confidence, persistence, no side effects."""

from __future__ import annotations

from pathlib import Path

from cortex.selfmodel import (
    CAPABILITY_ABSENT,
    CAPABILITY_PRESENT,
    SelfModel,
    SelfModelSnapshot,
)
from runtime.boot import boot_selfmodel
from substrate.format.substrate import Substrate


def test_never_succeeded_is_absent() -> None:
    view = SelfModel().query("GRASP", tick=0)

    assert view.status == CAPABILITY_ABSENT
    assert view.present is False
    assert view.confidence == 0.0
    assert view.successes == 0


def test_one_success_makes_a_capability_present() -> None:
    model = SelfModel()
    model.record("OBSERVE", True, "", 0)

    view = model.query("OBSERVE", tick=0)
    assert view.status == CAPABILITY_PRESENT
    assert view.successes == 1
    assert view.confidence > 0.0


def test_confidence_rises_with_repetition() -> None:
    once = SelfModel()
    once.record("OBSERVE", True, "", 0)
    many = SelfModel()
    for _ in range(5):
        many.record("OBSERVE", True, "", 0)

    assert many.query("OBSERVE", 0).confidence > once.query("OBSERVE", 0).confidence


def test_confidence_decays_with_recency() -> None:
    model = SelfModel()
    model.record("OBSERVE", True, "", 0)

    fresh = model.query("OBSERVE", tick=0).confidence
    later = model.query("OBSERVE", tick=100).confidence
    much_later = model.query("OBSERVE", tick=300).confidence

    assert fresh > later > much_later
    assert much_later > 0.0


def test_failures_alone_are_absent() -> None:
    model = SelfModel()
    model.record("GRASP", False, "slip-after-contact", 3)

    view = model.query("GRASP", tick=3)
    assert view.status == CAPABILITY_ABSENT
    assert view.failures == 1
    assert view.signature == "slip-after-contact"


def test_success_after_failure_is_present() -> None:
    model = SelfModel()
    model.record("GRASP", False, "slip-after-contact", 1)
    model.record("GRASP", True, "", 2)

    view = model.query("GRASP", tick=2)
    assert view.status == CAPABILITY_PRESENT
    assert view.successes == 1
    assert view.failures == 1


def test_attempts_are_persisted_to_the_substrate(tmp_path: Path) -> None:
    substrate = Substrate(tmp_path / "log")
    model = SelfModel(substrate=substrate)
    model.record("OBSERVE", True, "", 0)

    kinds = [record.kind for record in substrate.read_all()]
    assert "selfmodel" in kinds


def test_reboot_preserves_the_model(tmp_path: Path) -> None:
    substrate = Substrate(tmp_path / "log")
    model = SelfModel(substrate=substrate)
    model.record("OBSERVE", True, "", 0)
    model.record("OBSERVE", True, "", 1)
    model.record("GRASP", False, "slip-after-contact", 2)
    expected = model.snapshot(5).canonical_bytes()

    rebooted = boot_selfmodel(substrate)

    assert rebooted.snapshot(5).canonical_bytes() == expected
    assert rebooted.query("OBSERVE", 5).status == CAPABILITY_PRESENT
    assert rebooted.query("GRASP", 5).status == CAPABILITY_ABSENT


def test_query_has_no_side_effects(tmp_path: Path) -> None:
    substrate = Substrate(tmp_path / "log")
    model = SelfModel(substrate=substrate)
    model.record("OBSERVE", True, "", 0)
    before_records = len(substrate.read_all())
    before_state = model.to_dict(tick=2)

    first = model.query("OBSERVE", tick=2)
    second = model.query("OBSERVE", tick=2)

    assert first.to_dict() == second.to_dict()
    assert len(substrate.read_all()) == before_records
    assert model.to_dict(tick=2) == before_state


def test_snapshot_reports_unknown_verbs_as_absent() -> None:
    model = SelfModel()
    model.record("OBSERVE", True, "", 0)
    snapshot = model.snapshot(0)

    assert snapshot.query("OBSERVE").status == CAPABILITY_PRESENT
    assert snapshot.query("GRASP").status == CAPABILITY_ABSENT


def test_snapshot_round_trips() -> None:
    model = SelfModel()
    model.record("OBSERVE", True, "", 0)
    model.record("GRASP", False, "slip", 1)
    snapshot = model.snapshot(2)

    restored = SelfModelSnapshot.from_dict(snapshot.to_dict())

    assert restored.canonical_bytes() == snapshot.canonical_bytes()


def test_rebuilt_snapshot_continues_identically() -> None:
    model = SelfModel()
    model.record("OBSERVE", True, "", 0)
    model.record("OBSERVE", True, "", 1)
    rebuilt = SelfModel.from_dict(model.snapshot(2).to_dict())

    model.record("OBSERVE", True, "", 2)
    rebuilt.record("OBSERVE", True, "", 2)

    assert rebuilt.query("OBSERVE", 3).to_dict() == model.query("OBSERVE", 3).to_dict()
