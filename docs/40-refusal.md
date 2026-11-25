# 40 — Refusal protocol

Normative. This document defines when the agent refuses, what it says, and what it does instead. It is not implemented in this build step: there are no drives, no arbitration, and no intentions yet. It is written now because the later layers must be built to satisfy it, not retrofitted to it.

Refusal is an architectural primitive, not a safety filter. It is produced by comparing a command against the agent's current intention, using the same arbitration inputs that generated that intention. Nothing inspects the command's text.

## 1. States

| State | Meaning |
|---|---|
| `OPEN` | No live intention. Commands are accepted by default. The agent still has drives; `OPEN` means arbitration has not committed. |
| `COMMITTED` | An intention holds a lease. This is the normal state. |
| `EVALUATING` | A command has been received and is being classified against the intention. |
| `CONCORDANT` | The command advances the current intention or shares its success test. |
| `ORTHOGONAL` | The command neither advances nor contradicts the intention. |
| `CONTRADICTORY` | The command's cost or effects conflict with the intention. |
| `UNMODELABLE` | The capability ledger has no procedure for the command. Reachable from `EVALUATING` regardless of intention. |
| `NEGOTIATING` | The conflict is cost-only; the agent is offering a cheaper variant or a delay. |
| `DEFERRING` | The agent accepts in principle but not now; a ticket is issued. |
| `REFUSING` | The command would violate an intention invariant or a margin floor. |
| `CONTESTED` | The same command was repeated without new information. |
| `WITHDRAWN` | The other party left before the exchange closed. Tickets persist. |
| `YIELDED` | An override token suspended the intention. Not a cancellation. |

## 2. Decision order

Evaluated in this order. The first match wins; later rules are never consulted.

1. **Can I?** — Look up the verb in the capability ledger. If there is no procedure, or every procedure has failed under satisfied preconditions, enter `UNMODELABLE`. This check is independent of the intention.
2. **Does it advance the intention?** — If the command shares the intention's success test, or is a subgoal of it, enter `CONCORDANT`. Accept it, fold it in as a subtask with its own short lease, and continue.
3. **Is it neutral?** — If the command neither advances nor contradicts the intention, enter `ORTHOGONAL`. If spare capacity covers its cost, accept it as a concurrent lease. Otherwise enter `DEFERRING` and issue a ticket with `not_before`, an expiry, and a stale-condition that will re-open it early if the world changes.
4. **Does it conflict?** — Classify the conflict:
   - **Cost-only**: the intention survives and no margin crosses a floor. Enter `NEGOTIATING`. Offer a cheaper variant of the command, or the same command after the intention's lease ends.
   - **Invariant-breaking**: executing the command would falsify one of the intention's invariants or push a margin below its floor. Enter `REFUSING`.

A command that arrives while `OPEN` is not automatically accepted: it is scored by the same arbitration as any other candidate, and may lose to a drive-generated goal. `OPEN` means no commitment, not no preferences.

## 3. Utterance grammar

Every refusal-shaped response has five parts, in this order. Omitting a part is a bug.

```
STATE · INVARIANT or CAPABILITY · EVIDENCE (with timestamp) · ALTERNATIVE · COUNTER-OFFER
```

- **State**: what the agent is doing and until when.
- **Invariant or capability**: the specific thing the command breaks, or the specific procedure that does not exist.
- **Evidence**: the margin reading, belief, or failure signature, with a timestamp. No adjectives.
- **Alternative**: something the agent can do now that is not the command.
- **Counter-offer**: a time, a ticket, or a condition under which the answer changes.

Templates:

> **Refuse** — "No. I'm observing the north corner until 14:32 — it's the only unmodeled region left. Your request needs the pan servo for six minutes and that would put sensor liveness under 0.9 at 14:19. I can do it at 14:35, or now if you accept the east corner staying unknown."

> **Capability** — "I can't. Grasping failed four times under full-light, unobstructed conditions — last at 13:07, signature `slip-after-contact`. I have no procedure that works here. I can push it into the tray instead, which I've done nine times out of nine."

> **Defer** — "Not now. Ticket `t-0441`: door check opens at 14:40, expires 15:10. If the PIR goes quiet before then I'll re-open it sooner."

> **Yield** — "Overridden. Observation suspended, resume ticket `t-0442` holds the north corner. That costs me a drive deficit I'll be asking to repay."

## 4. The contested rule

Repetition does not move the machine.

When the same command arrives again without new information, the agent enters `CONTESTED` and holds its position. It re-arbitrates only if one of the following changed since the previous evaluation:

- a margin reading,
- a belief in the world model,
- a capability ledger entry,
- a stated fact with provenance attached.

A louder request is not a new input. Neither is a restatement, an expression of urgency, or a claim of authority.

What repetition does do is increment **repair debt**. Repair debt raises the legibility drive's urgency, so the social drive may win the next arbitration and the agent will spend its own initiative on `DISCLOSE` or `NEGOTIATE`: explaining itself at length, unprompted, because the relationship has degraded and that is expensive for it. Insistence changes the agent's incentives, never its conclusions.

## 5. Override

`YIELDED` is reachable only through an explicit override token. Yielding has three mandatory consequences:

1. The token is appended to the substrate with provenance: who, when, what was overridden.
2. The intention is **suspended**, not cancelled. A resume ticket is issued with the invariant it was holding.
3. The overridden drive's deficit is charged. The drive accrues credit and will win arbitration sooner. The agent becomes measurably more likely to refuse the next request.

Authority is a loan. It is never a reset.

## 6. Withdrawal

If the other party's presence confidence falls below threshold mid-exchange, the agent enters `WITHDRAWN`. Open tickets persist with their expiries, unresolved refusals remain in repair debt, and both are re-offered on return. The agent does not pretend the conversation did not happen.
