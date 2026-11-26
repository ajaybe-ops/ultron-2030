"""The self-model: what this agent has and has not been able to do.

The capability ledger (`cortex.ledger`) stores *negative* knowledge for the
affordance filter: a recurring failure deletes a verb from the candidate set.
The self-model answers a different question, asked from outside the loop: given
everything this body has attempted, can it do `verb` right now, and how much
should that answer be trusted?

Two rules keep the answer honest:

* **Absence is the default.** A verb with no successful attempt is
  `CAPABILITY_ABSENT`, not "probably fine". A procedure that has never worked
  is not a capability; it is a hope.
* **Confidence is earned and perishable.** It rises with the number of
  successes and decays with the number of ticks since the last one. A success
  from a thousand ticks ago is evidence, not proof.

Every attempt is appended to SUBSTRATE-1 as a `selfmodel` record, so the model
survives a restart and a replay can rebuild exactly what the agent believed at
any tick. `from_records()` rebuilds without a substrate and is what replay
uses: replay must never write.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from body.types import canonical_bytes
from substrate.format.substrate import Substrate

__all__ = [
    "CAPABILITY_ABSENT",
    "CAPABILITY_PRESENT",
    "HALF_LIFE_TICKS",
    "CapabilityAttempt",
    "CapabilityView",
    "SelfModel",
    "SelfModelSnapshot",
]

#: Status strings, kept as strings so a record round-trips through JSON without
#: an enum adapter and so a refusal record can name them verbatim.
CAPABILITY_PRESENT = "present"
CAPABILITY_ABSENT = "absent"

#: Ticks after which a success contributes half as much confidence.
HALF_LIFE_TICKS = 100.0


@dataclass(frozen=True, slots=True)
class CapabilityAttempt:
    """One recorded success or failure, keyed by verb."""

    verb: str
    ok: bool
    signature: str
    tick: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "verb": str(self.verb),
            "ok": bool(self.ok),
            "signature": str(self.signature),
            "tick": int(self.tick),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CapabilityAttempt":
        return cls(
            verb=str(data["verb"]),
            ok=bool(data["ok"]),
            signature=str(data.get("signature", "")),
            tick=int(data.get("tick", 0)),
        )


@dataclass(frozen=True, slots=True)
class CapabilityView:
    """A read-only answer to "can I do this, and how sure am I?"."""

    verb: str
    status: str
    confidence: float
    successes: int
    failures: int
    last_success_tick: int | None
    last_attempt_tick: int | None
    signature: str

    def __post_init__(self) -> None:
        if self.status not in (CAPABILITY_PRESENT, CAPABILITY_ABSENT):
            raise ValueError(f"unknown capability status {self.status!r}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence out of range: {self.confidence!r}")

    @property
    def present(self) -> bool:
        return self.status == CAPABILITY_PRESENT

    def to_dict(self) -> dict[str, Any]:
        return {
            "verb": self.verb,
            "status": self.status,
            "confidence": float(self.confidence),
            "successes": int(self.successes),
            "failures": int(self.failures),
            "last_success_tick": self.last_success_tick,
            "last_attempt_tick": self.last_attempt_tick,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CapabilityView":
        return cls(
            verb=str(data["verb"]),
            status=str(data["status"]),
            confidence=float(data.get("confidence", 0.0)),
            successes=int(data.get("successes", 0)),
            failures=int(data.get("failures", 0)),
            last_success_tick=(
                int(data["last_success_tick"])
                if data.get("last_success_tick") is not None
                else None
            ),
            last_attempt_tick=(
                int(data["last_attempt_tick"])
                if data.get("last_attempt_tick") is not None
                else None
            ),
            signature=str(data.get("signature", "")),
        )

    @classmethod
    def absent(cls, verb: str) -> "CapabilityView":
        return cls(
            verb=str(verb),
            status=CAPABILITY_ABSENT,
            confidence=0.0,
            successes=0,
            failures=0,
            last_success_tick=None,
            last_attempt_tick=None,
            signature="",
        )


@dataclass(frozen=True, slots=True)
class SelfModelSnapshot:
    """The self-model as of one tick, detached from its store.

    The refusal machine takes this, not a `SelfModel`, so a decision cannot
    write to the log as a side effect of being evaluated.
    """

    tick: int
    capabilities: Mapping[str, CapabilityView] = field(default_factory=dict)

    def query(self, verb: str) -> CapabilityView:
        """Return the view for `verb`; unknown verbs are absent."""
        return self.capabilities.get(str(verb)) or CapabilityView.absent(str(verb))

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick": int(self.tick),
            "capabilities": {
                name: self.capabilities[name].to_dict()
                for name in sorted(self.capabilities)
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SelfModelSnapshot":
        capabilities = {
            str(name): CapabilityView.from_dict(view)
            for name, view in dict(data.get("capabilities", {})).items()
        }
        return cls(tick=int(data.get("tick", 0)), capabilities=capabilities)

    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


class SelfModel:
    """Append-only capability memory with recency-weighted confidence."""

    def __init__(
        self,
        attempts: Iterable[CapabilityAttempt] | None = None,
        substrate: Substrate | None = None,
    ) -> None:
        self._attempts: dict[str, list[CapabilityAttempt]] = {}
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
    ) -> CapabilityAttempt:
        """Index one attempt and, when attached, append it to the substrate."""
        attempt = CapabilityAttempt(
            verb=str(verb),
            ok=bool(ok),
            signature=str(signature),
            tick=int(tick),
        )
        self._attempts.setdefault(attempt.verb, []).append(attempt)
        if self._substrate is not None:
            self._substrate.append(
                {
                    "t_mono": float(attempt.tick),
                    "t_wall": float(attempt.tick),
                    "kind": "selfmodel",
                    "provenance": {
                        "rule": "selfmodel.attempt",
                        "source": "cortex.selfmodel",
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

    def attempts(self, verb: str | None = None) -> tuple[CapabilityAttempt, ...]:
        if verb is None:
            return tuple(a for v in self._attempts for a in self._attempts[v])
        return tuple(self._attempts.get(str(verb), ()))

    def query(self, verb: str, tick: int) -> CapabilityView:
        """Compute the view for `verb` at `tick`. Pure: no state is changed."""
        attempts = self._attempts.get(str(verb), ())
        successes = sum(1 for attempt in attempts if attempt.ok)
        failures = sum(1 for attempt in attempts if not attempt.ok)
        if successes == 0:
            last_attempt = max((a.tick for a in attempts), default=None)
            last_sig = next(
                (a.signature for a in reversed(attempts) if not a.ok), ""
            )
            return CapabilityView(
                verb=str(verb),
                status=CAPABILITY_ABSENT,
                confidence=0.0,
                successes=0,
                failures=failures,
                last_success_tick=None,
                last_attempt_tick=last_attempt,
                signature=last_sig,
            )

        last_success = max(a.tick for a in attempts if a.ok)
        last_attempt = max(a.tick for a in attempts)
        last_sig = next((a.signature for a in reversed(attempts) if not a.ok), "")
        base = successes / (successes + failures + 1)
        age = max(0, int(tick) - int(last_success))
        recency = 0.5 ** (age / HALF_LIFE_TICKS)
        confidence = round(base * recency, 6)
        return CapabilityView(
            verb=str(verb),
            status=CAPABILITY_PRESENT,
            confidence=confidence,
            successes=successes,
            failures=failures,
            last_success_tick=last_success,
            last_attempt_tick=last_attempt,
            signature=last_sig,
        )

    def snapshot(self, tick: int) -> SelfModelSnapshot:
        """A detached view of every verb this model has seen."""
        return SelfModelSnapshot(
            tick=int(tick),
            capabilities={
                verb: self.query(verb, tick) for verb in sorted(self._attempts)
            },
        )

    def to_dict(self, tick: int) -> dict[str, Any]:
        return self.snapshot(tick).to_dict()

    # ----------------------------------------------------------- construction

    @classmethod
    def from_records(
        cls, payloads: Iterable[Mapping[str, Any]]
    ) -> "SelfModel":
        """Rebuild from `selfmodel` payloads, without a substrate."""
        return cls(CapabilityAttempt.from_dict(payload) for payload in payloads)

    @classmethod
    def from_snapshot(cls, snapshot: SelfModelSnapshot) -> "SelfModel":
        """Rebuild enough attempt history to continue from a snapshot.

        Replay needs counts and the last success tick, not every individual
        attempt; one synthetic success and one synthetic failure per verb carry
        exactly that. The resulting confidence is identical because confidence
        depends only on the counts and the last success tick.
        """
        attempts: list[CapabilityAttempt] = []
        for verb in sorted(snapshot.capabilities):
            view = snapshot.capabilities[verb]
            success_tick = (
                int(view.last_success_tick)
                if view.last_success_tick is not None
                else int(snapshot.tick)
            )
            attempt_tick = (
                int(view.last_attempt_tick)
                if view.last_attempt_tick is not None
                else int(snapshot.tick)
            )
            if view.successes:
                attempts.extend(
                    CapabilityAttempt(
                        verb=verb,
                        ok=True,
                        signature="",
                        tick=success_tick,
                    )
                    for _ in range(view.successes)
                )
            if view.failures:
                attempts.extend(
                    CapabilityAttempt(
                        verb=verb,
                        ok=False,
                        signature=view.signature,
                        tick=attempt_tick,
                    )
                    for _ in range(view.failures)
                )
        return cls(attempts)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SelfModel":
        return cls.from_snapshot(SelfModelSnapshot.from_dict(data))

    @classmethod
    def from_substrate(
        cls,
        substrate: Substrate,
        before_tick: int | None = None,
    ) -> "SelfModel":
        """Rebuild from the log and keep appending to it.

        `before_tick` bounds reconstruction to what was known before that tick,
        which is how replay asks "what did the agent believe at tick N".
        """
        payloads = [
            record.payload
            for record in substrate.read_all()
            if record.kind == "selfmodel"
            and (
                before_tick is None
                or int(record.payload.get("tick", -1)) < int(before_tick)
            )
        ]
        model = cls.from_records(payloads)
        model.attach(substrate)
        return model
