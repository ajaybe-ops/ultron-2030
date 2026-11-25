"""Arbitration: which drive gets the body this tick.

The engine is a pure competition over three candidate sets. It holds exactly
three things — the config, one `DriveState` per drive, and the live intention —
so the entire decision state is serialisable and therefore replayable.

The algorithm, in order:

1. Ask each drive for its urgency. Damp it by half inside the drive's
   refractory window, so a drive that was just served does not immediately win
   again.
2. Ask each drive for candidates. Drop anything below the value floor, anything
   the body does not declare or the ledger has learned is impossible, and
   anything whose cost would breach a margin floor.
3. If any margin is under its floor, restrict the field to D2 candidates. This
   is the VITAL override: viability is not one vote among three, it is the
   precondition for voting at all.
4. Score what is left. Ties break on `(-score, drive, verb, target)`, so the
   winner is a function of the inputs and nothing else.
5. If an intention is live, test success, then abort. If neither fires, keep
   it — unless the lease is up, a floor is breached, or VITAL has arrived, and
   the challenger clears `theta_preempt` times the incumbent's score.
6. Otherwise acquire the best candidate if it clears `theta_commit`. If
   nothing clears the threshold, DRIFT: take the cheapest high-information
   option for a single tick. If even that fails, do nothing and let the tick
   become REST.

Two documented choices:

* **The carry bonus is a shorter refractory window, not a stored goal.**
  Candidates are re-emitted every tick, so there is no goal object to carry a
  bonus on. A drive whose intention was aborted gets `refractory_ticks *
  carry_bonus` fewer ticks of damping, which is the observable meaning of
  "nearly-finished work is cheaper to restart".
* **A release and an acquisition can happen in the same tick.** Returning None
  on a completion would waste a tick; the engine re-arbitrates immediately so
  the body is never idle by accident.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from body.port import Body
from body.types import Margins, Snapshot
from cortex.arbitration.config import ArbitrationConfig
from cortex.arbitration.filters import affordance_filter, viability_filter
from cortex.arbitration.score import score
from cortex.drives import epistemic, homeostatic, legibility
from cortex.drives.state import DriveState, fresh_states
from cortex.goal import Goal
from cortex.intention.lease import Intention, intention_for_goal
from cortex.ledger import CapabilityLedger

__all__ = ["DRIVES", "Engine", "MIN_VALUE"]

#: Fixed order. Urgency and candidate order both depend on it.
DRIVES: tuple[tuple[str, Any], ...] = (
    ("D1", epistemic),
    ("D2", homeostatic),
    ("D3", legibility),
)

#: Candidates below this value are not worth scoring.
MIN_VALUE = 0.01

#: Refractory damping factor while a drive is inside its window.
REFRACTORY_DAMPING = 0.5


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class Engine:
    """The arbiter. Three fields, no hidden state."""

    def __init__(self, config: ArbitrationConfig | None = None) -> None:
        self.config = config if config is not None else ArbitrationConfig()
        self.drive_states: dict[str, DriveState] = fresh_states()
        self.current: Intention | None = None

    # ------------------------------------------------------------------- step

    def step(
        self,
        snapshot: Snapshot,
        margins: Margins,
        ledger: CapabilityLedger,
        tick: int,
        body: Body | None = None,
    ) -> Intention | None:
        """Run one arbitration round and return the intention to act on.

        `body` is used only for the affordance filter. When it is omitted the
        filter is skipped, which is how the scoring tests isolate a single
        variable; the live loop always passes a body.
        """
        config = self.config
        urgencies = self._urgencies(snapshot, margins, ledger, tick)
        candidates = self._candidates(snapshot, margins, ledger, body)
        ranked = self._rank(candidates, urgencies, tick)
        best_score, best_goal = ranked[0] if ranked else (0.0, None)

        current = self.current
        if current is not None:
            if current.check_success(tick, snapshot, margins):
                self._release(current, tick, completed=True)
                self.current = None
            elif current.check_abort(tick, snapshot, margins):
                self._release(current, tick, completed=False)
                self.current = None
            else:
                if self._should_preempt(
                    current, best_goal, best_score, urgencies, tick, margins
                ):
                    self._release(current, tick, completed=False)
                    return self._acquire(best_goal, tick)
                return current

        if best_goal is not None and best_score >= config.theta_commit:
            return self._acquire(best_goal, tick)

        return self._drift(candidates, urgencies, tick)

    # -------------------------------------------------------------- internals

    def _urgencies(
        self,
        snapshot: Snapshot,
        margins: Margins,
        ledger: CapabilityLedger,
        tick: int,
    ) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, module in DRIVES:
            state = self.drive_states[name]
            value = module.urgency(snapshot, margins, ledger, state)
            if tick < state.refractory_until:
                value *= REFRACTORY_DAMPING
            out[name] = _clamp01(value)
        return out

    def _candidates(
        self,
        snapshot: Snapshot,
        margins: Margins,
        ledger: CapabilityLedger,
        body: Body | None,
    ) -> list[Goal]:
        out: list[Goal] = []
        verdicts: dict[tuple[str, str], bool] = {}
        for name, module in DRIVES:
            for goal in module.candidates(
                snapshot, margins, ledger, self.drive_states[name]
            ):
                if goal.value < MIN_VALUE:
                    continue
                if body is not None:
                    key = (goal.drive, goal.verb)
                    if key not in verdicts:
                        verdicts[key] = affordance_filter(goal, ledger, body)
                    if not verdicts[key]:
                        continue
                if not viability_filter(goal, margins):
                    continue
                out.append(goal)

        if any(state.vital for state in self.drive_states.values()):
            restricted = [goal for goal in out if goal.drive == "D2"]
            if restricted:
                return restricted
        return out

    def _rank(
        self,
        candidates: Sequence[Goal],
        urgencies: dict[str, float],
        tick: int,
    ) -> list[tuple[float, Goal]]:
        scored = [
            (
                score(
                    goal,
                    urgencies[goal.drive],
                    self.drive_states[goal.drive],
                    tick,
                    self.config,
                ),
                goal,
            )
            for goal in candidates
        ]
        scored.sort(
            key=lambda pair: (
                -pair[0],
                pair[1].drive,
                pair[1].verb,
                pair[1].target or ("", -1),
            )
        )
        return scored

    def _should_preempt(
        self,
        current: Intention,
        best_goal: Goal | None,
        best_score: float,
        urgencies: dict[str, float],
        tick: int,
        margins: Margins,
    ) -> bool:
        if best_goal is None:
            return False

        vital = any(state.vital for state in self.drive_states.values())
        if vital and best_goal.drive == "D2" and current.goal.drive != "D2":
            return True

        if not (tick >= current.lease_until or vital or margins.below_floor()):
            return False

        incumbent = score(
            current.goal,
            urgencies[current.goal.drive],
            self.drive_states[current.goal.drive],
            tick,
            self.config,
        )
        return best_score >= self.config.theta_preempt * incumbent

    def _acquire(self, goal: Goal, tick: int, *, drift: bool = False) -> Intention:
        lease_ticks = 1 if drift else None
        intention = intention_for_goal(goal, tick, self.config, drift=drift,
                                       lease_ticks=lease_ticks)
        self.drive_states[goal.drive].last_served_tick = int(tick)
        self.current = intention
        return intention

    def _release(self, intention: Intention, tick: int, *, completed: bool) -> None:
        state = self.drive_states[intention.goal.drive]
        state.served_count += 1
        window = self.config.refractory_ticks
        if not completed:
            window -= int(round(window * self.config.carry_bonus))
        state.refractory_until = int(tick) + max(0, window)

    def _drift(
        self,
        candidates: Iterable[Goal],
        urgencies: dict[str, float],
        tick: int,
    ) -> Intention | None:
        """Take one cheap exploratory step when nothing clears the bar."""
        affordable = [
            goal
            for goal in candidates
            if goal.cost() <= self.config.drift_cost_cap
        ]
        if not affordable:
            self.current = None
            return None
        affordable.sort(
            key=lambda goal: (
                -(goal.expected_information_gain / (goal.cost() + 1e-9)),
                goal.drive,
                goal.verb,
                goal.target or ("", -1),
            )
        )
        return self._acquire(affordable[0], tick, drift=True)
