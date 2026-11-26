"""The refusal state machine: command in, decision out, record in the log.

`negotiation.refusal.decide()` is pure. This module owns the part that is not:
the transition from one state to the next, the response text, and the append of
the decision to SUBSTRATE-1. A refusal record is written with `kind="refusal"`,
which makes it replayable with the same harness that replays ticks.

The machine is deliberately small. It does not queue commands, hold a dialogue,
or infer intent from text. It evaluates one structured command at a time and
writes down what it decided and why.
"""

from __future__ import annotations

from typing import Iterable

from negotiation.refusal import decide
from negotiation.state import (
    Command,
    Decision,
    DrivePressure,
    RefusalDecision,
    RefusalRecord,
    RefusalState,
)
from substrate.format.substrate import Substrate

__all__ = ["RefusalMachine"]


class RefusalMachine:
    """Stateful wrapper around the pure decision function."""

    def __init__(self, substrate: Substrate | None = None) -> None:
        self.state = RefusalState.OPEN
        self.last: RefusalRecord | None = None
        self._substrate = substrate

    def attach(self, substrate: Substrate) -> None:
        self._substrate = substrate

    def receive(
        self,
        command: Command,
        intention,
        self_model_snapshot,
        drive_pressures: Iterable[DrivePressure],
    ) -> RefusalRecord:
        """Evaluate one command, respond, and persist the decision."""
        self.state = RefusalState.EVALUATING
        decision: RefusalDecision = decide(
            intention, command, self_model_snapshot, drive_pressures
        )
        self.state = decision.state
        self.last = decision.record
        if self._substrate is not None:
            self._substrate.append(
                {
                    "t_mono": float(command.tick),
                    "t_wall": float(command.tick),
                    "kind": "refusal",
                    "provenance": {
                        "rule": f"refusal.{decision.state.value.lower()}",
                        "source": "negotiation.machine",
                        "command": command.verb,
                    },
                    "payload": decision.record.to_dict(),
                }
            )
        return decision.record

    def respond(self, record: RefusalRecord | None = None) -> str:
        """The five-part utterance for the last (or supplied) decision."""
        chosen = record if record is not None else self.last
        if chosen is None:
            raise ValueError("no decision to respond to")
        return chosen.utterance()

    def records(self) -> tuple[RefusalRecord, ...]:
        """Every refusal record in the attached substrate, in log order."""
        if self._substrate is None:
            return ()
        return tuple(
            RefusalRecord.from_dict(record.payload)
            for record in self._substrate.read_all()
            if record.kind == "refusal"
        )

    @classmethod
    def from_substrate(cls, substrate: Substrate) -> "RefusalMachine":
        machine = cls(substrate)
        records = machine.records()
        if records:
            machine.last = records[-1]
            machine.state = RefusalState(records[-1].state)
        return machine
