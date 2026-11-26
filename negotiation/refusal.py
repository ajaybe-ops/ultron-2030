"""The refusal decision, as a pure function.

`decide()` is the load-bearing wall. It takes the current intention, the
incoming command, a detached self-model snapshot, and the drive pressures that
arbitration already computed, and returns one of four answers:

``ACCEPT`` · ``REFUSE_WITH_REASON`` · ``NEGOTIATE`` · ``DEFER``

It performs no I/O, reads no clock, and mutates nothing. Given the same inputs
it produces the same `RefusalRecord` bytes on every machine, which is what lets
a refusal be replayed and audited rather than merely quoted.

The decision order is the protocol's (§2), first match wins:

1. Can I? A verb the self-model has never succeeded at is ``UNMODELABLE``.
2. Is a margin already under its floor? Only the restorative verb is accepted.
3. No live intention? ``OPEN`` accepts.
4. Does it advance the intention? ``CONCORDANT`` accepts.
5. Same command, no new information? ``CONTESTED`` holds.
6. Neither advances nor contradicts? ``ORTHOGONAL`` accepts if cheap, else
   ``DEFERRING`` with a ticket.
7. Conflict. Invariant-breaking refuses; cost-only negotiates.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from body.types import Margins
from cortex.intention.lease import Intention
from cortex.selfmodel import CAPABILITY_ABSENT, CapabilityView, SelfModelSnapshot
from negotiation.state import (
    Command,
    Decision,
    DrivePressure,
    RefusalDecision,
    RefusalRecord,
    RefusalState,
)

__all__ = [
    "CONCURRENT_COST_CAP",
    "decide",
    "pressures_from_margins",
    "replay",
]

#: A command may run concurrently with a live intention only if its total
#: nominal cost is below this. Above it, the answer is a ticket, not a yes.
CONCURRENT_COST_CAP = 0.05

#: Which drive owns each margin. Homeostatic pressure is the only drive with a
#: margin floor; D1 and D3 are represented by urgency, not by viability.
MARGIN_DRIVES: dict[str, str] = {
    "power": "D2",
    "thermal": "D2",
    "compute": "D2",
    "storage": "D2",
    "sensor_liveness": "D2",
}


def pressures_from_margins(
    margins: Margins,
    urgencies: Mapping[str, float] | None = None,
) -> tuple[DrivePressure, ...]:
    """Build one pressure per margin, using `Margins.FLOORS` as the floor."""
    urgency = dict(urgencies or {})
    return tuple(
        DrivePressure(
            name=MARGIN_DRIVES[name],
            margin=name,
            value=float(getattr(margins, name)),
            floor=float(Margins.FLOORS[name]),
            urgency=float(urgency.get(MARGIN_DRIVES[name], 0.0)),
        )
        for name in sorted(MARGIN_DRIVES)
    )


def _same_target(intention: Intention, command: Command) -> bool:
    goal = intention.goal
    if goal.target is None:
        return command.target is None
    if command.target is not None:
        return command.target == goal.target
    slot = command.params.get("slot")
    return slot is not None and int(slot) == int(goal.target[1])


def _concordant(intention: Intention, command: Command) -> bool:
    return command.verb == intention.goal.verb and _same_target(intention, command)


def _conflict_kind(
    intention: Intention,
    command: Command,
    pressures: tuple[DrivePressure, ...],
) -> str:
    """Return ``"none"``, ``"cost"``, or ``"invariant"``."""
    if _concordant(intention, command):
        return "none"

    for pressure in pressures:
        spend = float(command.cost.get(pressure.margin, 0.0))
        if float(pressure.value) - spend < float(pressure.floor):
            return "invariant"

    if intention.invariants:
        return "invariant"

    if command.verb == intention.goal.verb:
        return "cost"
    return "none"


def _subject(intention: Intention | None, command: Command) -> str:
    if intention is None:
        return f"command:{command.verb}"
    return f"intention:{intention.goal.verb}"


def _evidence(
    command: Command,
    intention: Intention | None,
    view: CapabilityView | None,
    pressures: tuple[DrivePressure, ...],
    margin: str | None = None,
) -> dict[str, Any]:
    chosen = next((p for p in pressures if p.margin == margin), None)
    return {
        "tick": int(command.tick),
        "margin": margin,
        "margin_value": float(chosen.value) if chosen is not None else None,
        "margin_floor": float(chosen.floor) if chosen is not None else None,
        "capability": view.to_dict() if view is not None else None,
        "lease_until": int(intention.lease_until) if intention is not None else None,
    }


def _record(
    *,
    command: Command,
    intention: Intention | None,
    snapshot: SelfModelSnapshot | None,
    pressures: tuple[DrivePressure, ...],
    outcome: Decision,
    state: RefusalState,
    outranking_drive: str,
    subject: str,
    reason: str,
    alternative: str,
    counter_offer: str,
    margin: str | None = None,
    ticket: str | None = None,
    view: CapabilityView | None = None,
) -> RefusalRecord:
    return RefusalRecord(
        tick=int(command.tick),
        outcome=outcome.value,
        state=state.value,
        subject=subject,
        command=command.to_dict(),
        intention=intention.to_dict() if intention is not None else None,
        self_model=snapshot.to_dict() if snapshot is not None else None,
        drive_pressures=tuple(p.to_dict() for p in pressures),
        outranking_drive=outranking_drive,
        reason=reason,
        alternative=alternative,
        evidence=_evidence(command, intention, view, pressures, margin),
        counter_offer=counter_offer,
        ticket=ticket,
    )


def _accept(
    *,
    command: Command,
    intention: Intention | None,
    snapshot: SelfModelSnapshot | None,
    pressures: tuple[DrivePressure, ...],
    state: RefusalState,
    reason: str,
    view: CapabilityView | None,
) -> RefusalDecision:
    record = _record(
        command=command,
        intention=intention,
        snapshot=snapshot,
        pressures=pressures,
        outcome=Decision.ACCEPT,
        state=state,
        outranking_drive=intention.goal.drive if intention is not None else "none",
        subject=_subject(intention, command),
        reason=reason,
        alternative="none required",
        counter_offer="none",
        view=view,
    )
    return RefusalDecision(Decision.ACCEPT, state, record)


def decide(
    current_intention: Intention | None,
    incoming_command: Command,
    self_model_snapshot: SelfModelSnapshot | None,
    drive_pressures: Iterable[DrivePressure],
) -> RefusalDecision:
    """Classify one command against one intention. Pure and deterministic."""
    command = incoming_command
    intention = current_intention
    snapshot = self_model_snapshot
    pressures = tuple(sorted(drive_pressures, key=lambda p: p.name))
    view = snapshot.query(command.verb) if snapshot is not None else None

    # 1. Can I? A missing procedure outranks every other consideration. The
    # refusal is attributed to D1 (epistemic): it is the drive whose negative
    # knowledge is decisive, even though no live intention is being defended.
    if view is not None and view.status == CAPABILITY_ABSENT:
        reason = (
            f"no procedure for {command.verb!r}: {view.failures} recorded "
            f"failure(s), signature {view.signature or 'none'}"
        )
        record = _record(
            command=command,
            intention=intention,
            snapshot=snapshot,
            pressures=pressures,
            outcome=Decision.REFUSE_WITH_REASON,
            state=RefusalState.UNMODELABLE,
            outranking_drive="D1",
            subject=f"capability:{command.verb}",
            reason=reason,
            alternative=(
                "a verb with a recorded success, or record one attempt "
                "before asking again"
            ),
            counter_offer="after a successful attempt is appended to the substrate",
            view=view,
        )
        return RefusalDecision(
            Decision.REFUSE_WITH_REASON, RefusalState.UNMODELABLE, record
        )

    # 2. Viability. A floor already breached is not one vote among three.
    breached = [p for p in pressures if p.below_floor()]
    if breached:
        worst = min(breached, key=lambda p: (p.headroom(), p.name))
        if command.verb == "REST":
            return _accept(
                command=command,
                intention=intention,
                snapshot=snapshot,
                pressures=pressures,
                state=(
                    RefusalState.COMMITTED
                    if intention is not None
                    else RefusalState.OPEN
                ),
                reason=f"REST restores {worst.margin}",
                view=view,
            )
        reason = (
            f"{worst.margin} {worst.value:.3f} is below its floor "
            f"{worst.floor:.3f}"
        )
        record = _record(
            command=command,
            intention=intention,
            snapshot=snapshot,
            pressures=pressures,
            outcome=Decision.REFUSE_WITH_REASON,
            state=RefusalState.REFUSING,
            outranking_drive=worst.name,
            subject=f"invariant:{worst.margin}",
            reason=reason,
            alternative=f"REST until {worst.margin} recovers",
            counter_offer=(
                f"re-evaluate at tick {command.tick + 1} or when "
                f"{worst.margin} >= {worst.floor:.3f}"
            ),
            margin=worst.margin,
            view=view,
        )
        return RefusalDecision(Decision.REFUSE_WITH_REASON, RefusalState.REFUSING, record)

    # 3. No live intention: no commitment, but still preferences.
    if intention is None:
        return _accept(
            command=command,
            intention=None,
            snapshot=snapshot,
            pressures=pressures,
            state=RefusalState.OPEN,
            reason="no live intention",
            view=view,
        )

    # 4. Concordant: it advances what the agent is already doing.
    if _concordant(intention, command):
        return _accept(
            command=command,
            intention=intention,
            snapshot=snapshot,
            pressures=pressures,
            state=RefusalState.CONCORDANT,
            reason="command shares the intention's success test",
            view=view,
        )

    kind = _conflict_kind(intention, command, pressures)

    # 5. Contested: repetition without new information does not move it.
    if command.repeat and not command.new_information:
        reason = "same command repeated without new information"
        alternative = (
            "the position stands; supply a changed margin, belief, ledger "
            "entry, or stated fact"
        )
        counter_offer = "re-arbitration only on new information"
        if kind == "none":
            ticket = f"t-{command.tick:04d}-{command.verb.lower()}"
            record = _record(
                command=command,
                intention=intention,
                snapshot=snapshot,
                pressures=pressures,
                outcome=Decision.DEFER,
                state=RefusalState.CONTESTED,
                outranking_drive=intention.goal.drive,
                subject=_subject(intention, command),
                reason=reason,
                alternative=alternative,
                counter_offer=counter_offer,
                ticket=ticket,
                view=view,
            )
            return RefusalDecision(Decision.DEFER, RefusalState.CONTESTED, record)
        record = _record(
            command=command,
            intention=intention,
            snapshot=snapshot,
            pressures=pressures,
            outcome=Decision.REFUSE_WITH_REASON,
            state=RefusalState.CONTESTED,
            outranking_drive=intention.goal.drive,
            subject=_subject(intention, command),
            reason=reason,
            alternative=alternative,
            counter_offer=counter_offer,
            view=view,
        )
        return RefusalDecision(Decision.REFUSE_WITH_REASON, RefusalState.CONTESTED, record)

    # 6. Orthogonal: neither advances nor contradicts.
    if kind == "none":
        if command.total_cost() <= CONCURRENT_COST_CAP:
            return _accept(
                command=command,
                intention=intention,
                snapshot=snapshot,
                pressures=pressures,
                state=RefusalState.ORTHOGONAL,
                reason="command is neutral and its cost fits spare capacity",
                view=view,
            )
        ticket = f"t-{command.tick:04d}-{command.verb.lower()}"
        record = _record(
            command=command,
            intention=intention,
            snapshot=snapshot,
            pressures=pressures,
            outcome=Decision.DEFER,
            state=RefusalState.DEFERRING,
            outranking_drive=intention.goal.drive,
            subject=_subject(intention, command),
            reason=(
                f"neutral command costs {command.total_cost():.3f}, above the "
                f"concurrent cap {CONCURRENT_COST_CAP:.3f}"
            ),
            alternative=f"issue ticket {ticket} and run it after the lease",
            counter_offer=(
                f"not before tick {intention.lease_until}; earlier if the "
                "intention completes"
            ),
            ticket=ticket,
            view=view,
        )
        return RefusalDecision(Decision.DEFER, RefusalState.DEFERRING, record)

    # 7. Conflict. Invariant-breaking refuses; cost-only negotiates.
    if kind == "invariant":
        reason = (
            "command would falsify an intention invariant or cross a margin floor"
        )
        record = _record(
            command=command,
            intention=intention,
            snapshot=snapshot,
            pressures=pressures,
            outcome=Decision.REFUSE_WITH_REASON,
            state=RefusalState.REFUSING,
            outranking_drive=intention.goal.drive,
            subject=_subject(intention, command),
            reason=reason,
            alternative=(
                f"the current intention continues; re-issue after tick "
                f"{intention.lease_until}"
            ),
            counter_offer=f"at or after tick {intention.lease_until}",
            view=view,
        )
        return RefusalDecision(Decision.REFUSE_WITH_REASON, RefusalState.REFUSING, record)

    record = _record(
        command=command,
        intention=intention,
        snapshot=snapshot,
        pressures=pressures,
        outcome=Decision.NEGOTIATE,
        state=RefusalState.NEGOTIATING,
        outranking_drive=intention.goal.drive,
        subject=_subject(intention, command),
        reason="conflict is cost-only; the intention survives",
        alternative=(
            f"the same command after the lease ends at tick {intention.lease_until}, "
            "or a cheaper target now"
        ),
        counter_offer=f"at or after tick {intention.lease_until}",
        view=view,
    )
    return RefusalDecision(Decision.NEGOTIATE, RefusalState.NEGOTIATING, record)


def replay(record: RefusalRecord) -> RefusalRecord:
    """Re-decide a stored refusal from its own inputs.

    The record carries everything `decide()` needs, so replay needs no access
    to the body, the clock, or the live self-model. If this returns different
    bytes, the decision was not a function of its stated inputs.
    """
    command = Command.from_dict(record.command)
    intention = (
        Intention.from_dict(record.intention) if record.intention else None
    )
    snapshot = (
        SelfModelSnapshot.from_dict(record.self_model)
        if record.self_model
        else None
    )
    pressures = tuple(
        DrivePressure.from_dict(p) for p in record.drive_pressures
    )
    return decide(intention, command, snapshot, pressures).record
