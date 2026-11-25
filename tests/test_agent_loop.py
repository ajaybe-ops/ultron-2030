"""The load-bearing test: 200 ticks with nobody in the room, replayed exactly."""

from __future__ import annotations

from typing import Any

from lab.parity import assert_agent_parity
from lab.replay import Replay
from substrate.format.substrate import Substrate

EVENTS = {"acquired", "completed", "aborted", "preempted"}


def goals(payloads: list[dict[str, Any]]) -> list[tuple[str, str, Any]]:
    out: list[tuple[str, str, Any]] = []
    for payload in payloads:
        intention = payload["intention"]
        if intention is None:
            continue
        goal = intention["goal"]
        target = tuple(goal["target"]) if goal["target"] else None
        out.append((goal["drive"], goal["verb"], target))
    return out


def test_two_hundred_ticks_produce_distinct_intentions(
    agent_session: dict[str, Any],
) -> None:
    payloads = agent_session["payloads"]
    assert len(payloads) == 200
    distinct = set(goals(payloads))
    assert len(distinct) >= 3
    assert len(payloads[0]["intention_events"]) >= 1


def test_both_epistemic_and_homeostatic_drives_win(
    agent_session: dict[str, Any],
) -> None:
    drives = {drive for drive, _, _ in goals(agent_session["payloads"])}
    assert {"D1", "D2"}.issubset(drives)


def test_two_hundred_tick_replay_is_byte_identical(
    agent_session: dict[str, Any],
) -> None:
    assert assert_agent_parity(agent_session["path"]) == 200


def test_replay_compares_the_whole_payload_not_just_the_snapshot(
    agent_session: dict[str, Any],
) -> None:
    replay = Replay(agent_session["path"])
    assert replay.is_agent is True
    assert replay.run() == replay.recorded_bytes()


def test_intention_events_are_well_formed(agent_session: dict[str, Any]) -> None:
    for payload in agent_session["payloads"]:
        events = payload["intention_events"]
        assert len(events) <= 2
        for event in events:
            assert event["event"] in EVENTS
            assert event["tick"] == payload["tick"]
            assert set(event) == {
                "event",
                "tick",
                "drive",
                "verb",
                "target",
                "lease_until",
                "drift",
            }


def test_agent_session_keeps_the_log_gap_free(
    agent_session: dict[str, Any],
) -> None:
    records = Substrate(agent_session["path"]).read_all()
    seqs = [record.seq for record in records]
    assert seqs == list(range(1, len(seqs) + 1))
