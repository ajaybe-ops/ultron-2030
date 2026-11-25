"""The cortex: drives, arbitration, intentions, and the capability ledger.

This package is the agent's decision layer. It reads `Snapshot`, `Margins`, and
the ledger, and it writes intentions. It never touches a body directly; the
only way out is `Intention.next_action()`.
"""

from cortex.goal import Goal
from cortex.ledger import CapabilityLedger, FailureSignature

__all__ = ["CapabilityLedger", "FailureSignature", "Goal"]
