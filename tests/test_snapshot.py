"""Snapshots: absence is represented, residuals are always present."""

from __future__ import annotations

import pytest

from body.embodiments.gridworld import PROXIMITY_SLOT, SLOT_COUNT, GridWorld
from body.types import Snapshot


def test_dead_sensor_decays_instead_of_disappearing() -> None:
    body = GridWorld()
    body.sense()
    body.disable_sensor()

    assert body.margins().sensor_liveness == 0.0
    first = body.sense().cell("proximity", PROXIMITY_SLOT)
    body.actuate({"verb": "REST"})
    second = body.sense().cell("proximity", PROXIMITY_SLOT)

    assert second.freshness < first.freshness
    assert second.var > first.var
    assert second.observed_at == first.observed_at == 0.0
    assert second.value == first.value


def test_residual_is_present_and_non_zero_after_motion() -> None:
    body = GridWorld(seed=5)
    body.actuate({"verb": "MOVE", "dx": 1, "dy": 0})
    snapshot = body.sense()

    assert len(snapshot.residual) == SLOT_COUNT
    assert all(isinstance(r, float) for r in snapshot.residual)
    assert max(snapshot.residual) > 0.1
    assert min(snapshot.residual) >= 0.0


def test_absence_is_a_value_never_a_hole() -> None:
    snapshot = GridWorld().sense()
    unseen = snapshot.cells[99]

    assert unseen.value == pytest.approx(0.5)
    assert unseen.var == pytest.approx(0.25)
    assert unseen.freshness == 0.0
    assert unseen.observed_at == 0.0
    assert all(cell is not None for cell in snapshot.cells)


def test_canonical_bytes_are_stable_across_round_trip() -> None:
    snapshot = GridWorld(seed=8).sense()
    restored = Snapshot.from_dict(snapshot.to_dict())

    assert snapshot.canonical_bytes() == restored.canonical_bytes()
    assert snapshot.digest() == restored.digest()


def test_misaligned_residual_is_rejected() -> None:
    snapshot = GridWorld().sense()
    with pytest.raises(ValueError, match="residual"):
        Snapshot(
            t_mono=snapshot.t_mono,
            t_wall=snapshot.t_wall,
            seq=snapshot.seq,
            body_id=snapshot.body_id,
            cells=snapshot.cells,
            residual=snapshot.residual[:-1],
        )
