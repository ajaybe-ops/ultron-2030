"""The Body port: the only surface through which the agent core touches matter.

Architectural invariant (dossier §2): the core reads exactly three things —
`Snapshot`, `Margins`, and the capability ledger — and it reaches the world
through exactly one thing, this port. A body that implements these four
methods is interchangeable. `body/embodiments/gridworld.py` and
`body/embodiments/rig_pi/` are the two implementations; neither leaks its
hardware into the core.

The port is deliberately four methods wide. Anything else a body wants to
offer (state serialisation, calibration, sensor enable/disable) is body-local
and may not be assumed by the core. `lab/replay` is allowed to reach past the
port because it reconstructs a specific body; the core is not.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

from body.types import Affordance, ActionResult, Margins, Snapshot

__all__ = ["Body"]


class Body(ABC):
    """Abstract embodiment. Four methods, no more.

    Implementations must be deterministic given their internal state: replay
    depends on it, and a body that samples from an unseeded RNG cannot be
    re-executed. Any stochasticity must be a pure function of `(seed, tick,
    slot)` or be part of the explicitly serialised state.
    """

    @abstractmethod
    def sense(self) -> Snapshot:
        """Return the current fused snapshot and its residual.

        Must be idempotent at a fixed clock reading: two calls with no
        intervening `actuate` return equal snapshots. Absent modalities are
        represented by stale cells, never by missing ones.
        """

    @abstractmethod
    def actuate(self, action: Mapping[str, Any]) -> ActionResult:
        """Attempt an action and report the outcome and its margin cost.

        A body may refuse physically (blocked, out of range) and reports that
        as `ok=False` or an explanatory `detail`; it must not raise for a
        well-formed but impossible action, because failure is data the
        self-model needs.
        """

    @abstractmethod
    def declare_affordances(self) -> Sequence[Affordance]:
        """Return the verbs this body can currently perform, with parameters."""

    @abstractmethod
    def margins(self) -> Margins:
        """Return current viability headroom on every axis."""
