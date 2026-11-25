"""Leases: a goal becomes an intention only when it is bounded in time.

An intention is the unit of commitment. It carries the goal as data, a lease
deadline, a set of invariants that must keep holding, and two named tests —
success and abort — that are stored as specs, not closures, so the whole thing
can be written to SUBSTRATE-1 and rebuilt after a restart.

Three properties matter:

* A lease always expires. Nothing is pursued forever because it once scored
  well.
* Success and abort are different events. Success resets the drive's
  refractory window; abort shortens it, because work that was nearly done is
  cheaper to restart than work that was never started.
* Every field round-trips. `from_dict(to_dict(x)) == x`, including the tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from body.types import Margins, Snapshot
from cortex.goal import Goal
from cortex.intention import registry

__all__ = ["Intention", "intention_for_goal"]

#: Lease lengths, in ticks, per verb. A verb not listed uses `goal.latency`.
DEFAULT_LEASES: dict[str, int] = {
    "OBSERVE": 2,
    "PROBE": 3,
    "REST": 3,
}


@dataclass(slots=True)
class Intention:
    """A goal under lease, with the tests that end the lease."""

    goal: Goal
    lease_until: int
    invariants: tuple[Mapping[str, Any], ...] = ()
    success_spec: Mapping[str, Any] | None = None
    abort_spec: Mapping[str, Any] | None = None
    resume_ticket: str | None = None
    acquired_tick: int = 0
    drift: bool = False

    # ------------------------------------------------------------------ tests

    def check_success(self, tick: int, snapshot: Snapshot, margins: Margins) -> bool:
        """True when the goal's success condition holds right now."""
        return registry.evaluate(self.success_spec, tick, snapshot, margins)

    def check_abort(self, tick: int, snapshot: Snapshot, margins: Margins) -> bool:
        """True when the abort test fires or any invariant has been broken."""
        if registry.evaluate(self.abort_spec, tick, snapshot, margins):
            return True
        return any(
            not registry.evaluate(spec, tick, snapshot, margins)
            for spec in self.invariants
        )

    # ----------------------------------------------------------------- action

    def next_action(
        self, snapshot: Snapshot, margins: Margins
    ) -> dict[str, Any]:
        """Translate the goal into a body action; see `actions.next_action`."""
        from cortex.intention import actions

        return actions.next_action(self, snapshot, margins)

    # ----------------------------------------------------------- persistence

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal.to_dict(),
            "lease_until": int(self.lease_until),
            "invariants": [dict(spec) for spec in self.invariants],
            "success_spec": dict(self.success_spec) if self.success_spec else None,
            "abort_spec": dict(self.abort_spec) if self.abort_spec else None,
            "resume_ticket": self.resume_ticket,
            "acquired_tick": int(self.acquired_tick),
            "drift": bool(self.drift),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Intention":
        success = data.get("success_spec")
        abort = data.get("abort_spec")
        return cls(
            goal=Goal.from_dict(data["goal"]),
            lease_until=int(data["lease_until"]),
            invariants=tuple(dict(spec) for spec in data.get("invariants", ())),
            success_spec=dict(success) if success else None,
            abort_spec=dict(abort) if abort else None,
            resume_ticket=data.get("resume_ticket"),
            acquired_tick=int(data.get("acquired_tick", 0)),
            drift=bool(data.get("drift", False)),
        )


def intention_for_goal(
    goal: Goal,
    tick: int,
    config: object | None = None,
    *,
    drift: bool = False,
    lease_ticks: int | None = None,
) -> Intention:
    """Wrap a goal in a lease with the tests appropriate to its verb.

    `config` is accepted for signature stability with the arbitration layer;
    the lease lengths are structural, not tunable, so they live here. `drift`
    marks a one-tick exploratory lease taken when nothing cleared the commit
    threshold.
    """
    del config
    start = int(tick)
    length = int(lease_ticks) if lease_ticks is not None else DEFAULT_LEASES.get(
        goal.verb, max(1, int(goal.latency))
    )
    lease_until = start + max(1, length)

    success: dict[str, Any] | None = None
    if goal.verb == "OBSERVE" and goal.target is not None:
        frame, slot = goal.target
        success = registry.make(
            "cell_fresh", frame=frame, slot=int(slot), threshold=0.9
        )
    elif goal.verb == "PROBE" and goal.target is not None:
        frame, slot = goal.target
        success = registry.make(
            "cell_var_below", frame=frame, slot=int(slot), threshold=0.2
        )
    elif goal.verb == "REST":
        success = registry.make("margin_recovered", name="power", threshold=0.9)

    abort = registry.make("lease_expired", lease_until=lease_until)
    return Intention(
        goal=goal,
        lease_until=lease_until,
        invariants=(),
        success_spec=success,
        abort_spec=abort,
        resume_ticket=None,
        acquired_tick=start,
        drift=bool(drift),
    )
