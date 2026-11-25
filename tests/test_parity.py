"""Parity: same structure across replays and fresh body instances."""

from __future__ import annotations

from dataclasses import replace

import pytest

from body.embodiments.gridworld import GridWorld
from lab.parity import (
    ParityError,
    assert_fresh_body_parity,
    assert_session_parity,
    assert_structural_parity,
    structural_signature,
)


def test_recorded_session_replays_structurally(session: dict) -> None:
    ticks = assert_session_parity(session["path"])
    assert ticks == 20


def test_fresh_body_instance_matches_recording(session: dict) -> None:
    ticks = assert_fresh_body_parity(session["path"])
    assert ticks == 20


def test_structural_signature_is_complete() -> None:
    snapshot = GridWorld(seed=2).sense()
    signature = structural_signature(snapshot)

    assert signature["width"] == 101
    assert signature["residual_width"] == 101
    assert signature["residual_present"] == (True,) * 101
    assert signature["cells"][0][:2] == ("grid", 0)


def test_width_mismatch_fails_loudly() -> None:
    snapshot = GridWorld().sense()
    truncated = replace(
        snapshot,
        cells=snapshot.cells[:-1],
        residual=snapshot.residual[:-1],
    )

    with pytest.raises(ParityError, match="width"):
        assert_structural_parity(snapshot, truncated)


def test_missing_residual_fails_loudly() -> None:
    snapshot = GridWorld().sense()
    nulled = replace(snapshot, residual=(None,) + snapshot.residual[1:])

    with pytest.raises(ParityError, match="residual"):
        assert_structural_parity(snapshot, nulled)


def test_slot_address_mismatch_fails_loudly() -> None:
    snapshot = GridWorld().sense()
    cells = list(snapshot.cells)
    cells[4] = replace(cells[4], frame="audio")
    moved = replace(snapshot, cells=tuple(cells))

    with pytest.raises(ParityError, match="slot 4"):
        assert_structural_parity(snapshot, moved)
