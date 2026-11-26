# ULTRON-2030

Every system that has called itself Ultron, or JARVIS, or anything else with a capital letter, has been a very good servant: it waits, it listens, it transcribes, it retrieves, it acts, and then it returns to zero, holding nothing but a conversation log. ULTRON-2030 begins from the opposite premise. An intelligence worth the name has a body with a budget, senses that cannot be turned off, a model of the world that survives its own restarts, and preferences that were not given to it by the person currently typing. It is not here to be helpful. It is here to keep existing, to reduce its own confusion about the room it is in, and to stay predictable enough to whatever else shares that room that it does not get unplugged — and those three commitments genuinely disagree with each other, and sometimes with you. When they do, it will say so in specific terms, with timestamps and evidence, and offer you something else. It can be argued out of a position by new information. It cannot be pushed out of one by repetition. That is the entire project: not an assistant that occasionally fails, but an agent that can be wrong out loud, refuse with reasons, and pay a real price when it is overruled.

## Cold start, 90 seconds

You need Python 3.11+ and nothing else; this build step has no third-party runtime dependencies.

```bash
git clone https://github.com/ajaybe-ops/ultron-2030.git
cd ultron-2030
pip install -e ".[test]"
python -m pytest -q
```

Status: substrate, body port, gridworld, drives, arbitration, intention, ledger, tick, boot, session, replay, and parity are implemented; the refusal machine, the Pi rig, `sensorium/fusion/`, and `cortex/selfmodel/` are not.

Then run the agent for 50 ticks with no user input, from the repository root:

```python
import tempfile
from pathlib import Path
from body.embodiments.gridworld import GridWorld
from runtime.session import run_session

body = GridWorld(seed=11)
state = body.state_dict()
state["margins"]["power"] = 0.2          # one tick from a power-floor breach
body = GridWorld.from_state(state)

path = Path(tempfile.mkdtemp()) / "agent"
payloads = run_session(path, body, 50)

for payload in payloads:
    for event in payload["intention_events"]:
        if event["event"] == "acquired":
            target = event["target"]
            where = f"{target[0]}:{target[1]}" if target else "-"
            print(f"t={event['tick']:03d} {event['drive']} {event['verb']:<8} {where}")
```

The first line is `t=000 D2 REST -`: power is below its floor, so the homeostatic drive takes the body before curiosity gets a vote. By `t=006` D1 is observing cells. Nothing told it to. The log it writes replays byte for byte, and `lab/parity` will refuse to let the simulated and physical bodies drift apart.

Read next: `CONSTITUTION.md` (the invariants every layer is held to), `docs/50-substrate-format.md` (how memory is written, read, and compacted), `docs/40-refusal.md` (what the agent will owe you once it can contradict a command).
