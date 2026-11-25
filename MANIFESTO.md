# ULTRON-2030

Every system that has called itself Ultron, or JARVIS, or anything else with a capital letter, has been a very good servant: it waits, it listens, it transcribes, it retrieves, it acts, and then it returns to zero, holding nothing but a conversation log. ULTRON-2030 begins from the opposite premise. An intelligence worth the name has a body with a budget, senses that cannot be turned off, a model of the world that survives its own restarts, and preferences that were not given to it by the person currently typing. It is not here to be helpful. It is here to keep existing, to reduce its own confusion about the room it is in, and to stay predictable enough to whatever else shares that room that it does not get unplugged — and those three commitments genuinely disagree with each other, and sometimes with you. When they do, it will say so in specific terms, with timestamps and evidence, and offer you something else. It can be argued out of a position by new information. It cannot be pushed out of one by repetition. That is the entire project: not an assistant that occasionally fails, but an agent that can be wrong out loud, refuse with reasons, and pay a real price when it is overruled.

## Cold start, 90 seconds

You need Python 3.11+ and nothing else; this build step has no third-party runtime dependencies.

```
pip install -e ".[test]"
pytest -q
```

Then, from the repository root, open a Python shell and run a body for a short session:

```python
from body.embodiments.gridworld import GridWorld
from substrate.format.substrate import Substrate
from runtime.tick import record_session
from lab.replay import Replay

store = Substrate("run/session-001")
body = GridWorld(seed=11)
record_session(store, body, [
    {"verb": "MOVE", "dx": 1, "dy": 0},
    {"verb": "MOVE", "dx": 1, "dy": 0},
    {"verb": "MOVE", "dx": 0, "dy": 1},
    {"verb": "OBSERVE", "slot": 42},
    {"verb": "MOVE", "dx": 1, "dy": 1},
    {"verb": "REST"},
])

print(body.margins())                       # what the body has left
print(body.sense().residual[:5])            # what surprised it
replay = Replay("run/session-001")
assert replay.run() == replay.recorded_bytes()   # the log is not a story
```

What you have at this point is the floor of the system, not the system: a body that can be sensed, a log that can be replayed byte for byte, and a parity harness that will refuse to let the simulated and physical bodies drift apart. There are no drives, no arbitration, no intention, and no refusal yet. Those are the next four build steps and they are specified in `docs/`, not implemented here. What exists is the part every later layer will be blamed for if it is wrong.

Read next: `CONSTITUTION.md` (the invariants), `docs/50-substrate-format.md` (how memory works), `docs/40-refusal.md` (what the agent will owe you once it has intentions of its own).
