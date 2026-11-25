"""Replay: byte-identical re-execution, and loud divergence."""

from __future__ import annotations

from pathlib import Path

import pytest

from lab.replay import Replay, ReplayDivergence, ReplayError
from substrate.format.substrate import Substrate

from conftest import tamper_tick_snapshot


def test_replay_is_byte_identical(session: dict) -> None:
    replay = Replay(session["path"])
    assert len(replay.ticks) == 20
    assert replay.run() == replay.recorded_bytes()
    replay.verify()


def test_replay_of_subset_matches_subset(session: dict) -> None:
    replay = Replay(session["path"], tick_range=(5, 9))
    produced = replay.run()
    assert produced == replay.recorded_bytes()
    assert len(replay.recorded_snapshots()) == 5
    assert produced.count(b"\n") == 4


def test_divergence_names_first_tick_and_field(session: dict) -> None:
    tamper_tick_snapshot(session["path"], tick_index=7, slot=3)

    with pytest.raises(ReplayDivergence) as caught:
        Replay(session["path"]).run()

    error = caught.value
    assert error.tick_index == 7
    assert error.field.startswith("cells[3]")
    assert "tick 7 diverged" in str(error)


def test_divergence_does_not_cascade(session: dict) -> None:
    """A lie at tick 7 must be reported at 7, not at 8 through body state."""
    tamper_tick_snapshot(session["path"], tick_index=7, slot=40, value=0.999)
    with pytest.raises(ReplayDivergence) as caught:
        Replay(session["path"]).run()
    assert caught.value.tick_index == 7


def test_missing_session_is_rejected(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    Substrate(empty)
    with pytest.raises(ReplayError):
        Replay(empty)


def test_out_of_range_tick_window_is_rejected(session: dict) -> None:
    with pytest.raises(ReplayError):
        Replay(session["path"], tick_range=(10, 99))
