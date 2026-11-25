"""Capability ledger: the agent's memory of what it cannot do.

Positive knowledge is cheap — a successful action leaves a mark in the world.
Negative knowledge is what has to be stored, because a body that keeps trying
the same impossible thing is not curious, it is broken. Every attempt is
appended to SUBSTRATE-1 as a `ledger` record; `recurring_failure()` folds them
into a signature, and the affordance filter uses that signature to remove a
verb from the candidate set entirely. The ledger does not lower a score. It
deletes the option.

A signature is the body's own `detail` string (`BLOCKED`, `OUT_OF_RANGE`,
`UNKNOWN_VERB`). Two failures with the same signature under the same satisfied
preconditions is the threshold for "recurring"; one failure is an accident.
A later success clears the count, because a body that has since learned to do
the thing is no longer incapable of it.

`from_substrate()` rebuilds the ledger for warm start and attaches it, so new
attempts continue to append. `from_records()` rebuilds without a substrate and
is what replay uses: replay must never write.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from substrate.format.substrate import Substrate

__all__ = ["Attempt", "CapabilityLedger", "FailureSignature", "RECURRING_THRESHOLD"]

#: How many failures with one signature make a capability "known broken".
RECURRING_THRESHOLD = 2


@dataclass(frozen=True, slots=True)
class Attempt:
    """One recorded attempt, successful or not."""

    verb: str
    ok: bool
    signature: str
    tick: int
    preconditions: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verb": str(self.verb),
            "ok": bool(self.ok),
            "signature": str(self.signature),
            "tick": int(self.tick),
            "preconditions": dict(self.preconditions),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Attempt":
        return cls(
            verb=str(data["verb"]),
            ok=bool(data["ok"]),
            signature=str(data.get("signature", "")),
            tick=int(data.get("tick", 0)),
            preconditions=dict(data.get("preconditions", {})),
        )


@dataclass(frozen=True, slots=True)
class FailureSignature:
    """A recurring failure: what it looked like, when, and how often."""

    signature: str
    tick: int
    count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "signature": self.signature,
            "tick": int(self.tick),
            "count": int(self.count),
        }


def _matches(attempt: Attempt, query: Mapping[str, Any]) -> bool:
    """An attempt matches a query when every queried precondition agrees."""
    return all(attempt.preconditions.get(key) == value for key, value in query.items())


class CapabilityLedger:
    """Append-only negative knowledge, keyed by verb.

    Instances are cheap to build and cheap to copy; the substrate is the only
    durable store. The in-memory index exists so the arbitration loop does not
    read the log on every tick.
    """

    def __init__(
        self,
        attempts: Iterable[Attempt] | None = None,
        substrate: Substrate | None = None,
    ) -> None:
        self._attempts: dict[str, list[Attempt]] = {}
        self._substrate = substrate
        for attempt in attempts or ():
            self._attempts.setdefault(attempt.verb, []).append(attempt)

    # ------------------------------------------------------------------ write

    def record(
        self,
        verb: str,
        ok: bool,
        signature: str,
        tick: int,
        preconditions: Mapping[str, Any] | None = None,
    ) -> Attempt:
        """Index one attempt and, when attached, append it to the substrate."""
        attempt = Attempt(
            verb=str(verb),
            ok=bool(ok),
            signature=str(signature),
            tick=int(tick),
            preconditions=dict(preconditions or {}),
        )
        self._attempts.setdefault(attempt.verb, []).append(attempt)
        if self._substrate is not None:
            self._substrate.append(
                {
                    "t_mono": float(attempt.tick),
                    "t_wall": float(attempt.tick),
                    "kind": "ledger",
                    "provenance": {
                        "rule": "ledger.attempt",
                        "source": "cortex.ledger",
                        "verb": attempt.verb,
                    },
                    "payload": attempt.to_dict(),
                }
            )
        return attempt

    def attach(self, substrate: Substrate) -> None:
        """Start appending new attempts to `substrate`."""
        self._substrate = substrate

    # ------------------------------------------------------------------- read

    def attempts(self, verb: str | None = None) -> tuple[Attempt, ...]:
        """Recorded attempts, in log order, optionally for one verb."""
        if verb is None:
            return tuple(a for v in self._attempts for a in self._attempts[v])
        return tuple(self._attempts.get(str(verb), ()))

    def can(self, verb: str, preconditions: Mapping[str, Any] | None = None) -> bool:
        """True unless this verb is a known dead end under these conditions."""
        return self.recurring_failure(verb, preconditions) is None

    def recurring_failure(
        self,
        verb: str,
        preconditions: Mapping[str, Any] | None = None,
    ) -> FailureSignature | None:
        """Return the most recent recurring failure, or None.

        A success clears the running count for that verb, so the signature that
        is returned always describes failures that happened *after* the last
        time the verb worked.
        """
        query = dict(preconditions or {})
        counts: dict[str, int] = {}
        last: Attempt | None = None
        for attempt in self._attempts.get(str(verb), ()):
            if not _matches(attempt, query):
                continue
            if attempt.ok:
                counts.clear()
                last = None
                continue
            counts[attempt.signature] = counts.get(attempt.signature, 0) + 1
            last = attempt
        if last is None:
            return None
        count = counts.get(last.signature, 0)
        if count < RECURRING_THRESHOLD:
            return None
        return FailureSignature(signature=last.signature, tick=last.tick, count=count)

    # ----------------------------------------------------------- construction

    @classmethod
    def from_records(cls, payloads: Iterable[Mapping[str, Any]]) -> "CapabilityLedger":
        """Rebuild from ledger payloads in `seq` order, without a substrate."""
        return cls(Attempt.from_dict(payload) for payload in payloads)

    @classmethod
    def from_substrate(
        cls,
        substrate: Substrate,
        before_seq: int | None = None,
    ) -> "CapabilityLedger":
        """Rebuild from the substrate and keep appending to it.

        `before_seq` bounds the reconstruction to what was known at that point,
        which is how replay asks "what did the ledger look like before tick N".
        """
        payloads = [
            record.payload
            for record in substrate.read_all()
            if record.kind == "ledger"
            and (before_seq is None or record.seq < int(before_seq))
        ]
        ledger = cls.from_records(payloads)
        ledger.attach(substrate)
        return ledger
