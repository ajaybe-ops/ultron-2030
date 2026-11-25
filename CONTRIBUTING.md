# Contributing to ULTRON-2030

The rules here are enforced by the test suite more than by review. Read them
before opening a pull request; they are short and they do not bend.

## Ground rules

1. **Standard library and pytest only.** No runtime dependencies. If a change
   needs a package, the change is wrong for this build step.
2. **Every new behaviour gets a test.** A behaviour without a test is a claim
   without evidence, and this repository does not ship claims.
3. **Replay must stay byte-identical.** Run `python -m pytest -q`. The 200-tick
   agent session and its replay are part of the suite. If they fail, fix the
   change, not the test.
4. **Every module has a docstring** that says why it exists, not what its
   functions are named. No TODOs and no commented-out code.
5. **No new abstractions without a home.** If a shared type is genuinely
   missing, add it to `body/types.py` with a docstring explaining why it could
   not live where it was first needed.
6. **Do not edit `MANIFESTO.md`, `CONSTITUTION.md`, or the normative files in
   `docs/`** to make a test pass. If the code and the document disagree, say so
   in the pull request.
7. **Do not change the SUBSTRATE-1 on-disk format** without a new format string
   and a migration note. Old logs must remain readable.

## Running the checks

```bash
python -m pytest -q
```

To verify a specific recorded session:

```python
from lab.parity import assert_agent_parity
assert_agent_parity("run/agent-001")
```

## Adding a drive

1. Create `cortex/drives/<name>.py` with `urgency(snapshot, margins, ledger,
   state)` and `candidates(snapshot, margins, ledger, state)`.
2. Add any persistent memory to `DriveState` in `cortex/drives/state.py`, with
   `to_dict`/`from_dict`. Unserialised memory cannot be replayed.
3. Append the module to `DRIVES` in `cortex/arbitration/engine.py` and add its
   key to `fresh_states()`.
4. Add tests in `tests/test_drives.py` for the urgency formula on a hand-built
   snapshot, and for the candidate set.

A drive that has nothing to say is allowed. D3 is quiet in the gridworld and
still runs every tick; that is the design, not a gap.

## Adding a body

1. Subclass `body.port.Body` and implement `sense`, `actuate`,
   `declare_affordances`, and `margins`.
2. Make `state_dict()` and `from_state()` exact. Replay reconstructs the body
   from the previous tick's state and nothing else.
3. Register the class in `DEFAULT_BODIES` in `lab/replay/__init__.py`.
4. Do not import the body from the core. If a core module needs to know the
   class name, the change is in the wrong layer.

## Reporting a replay divergence

Include the tick index, the field path, and the expected and actual values.
`ReplayDivergence` carries all four. A divergence is a bug in the code that
introduced it, never in the harness.

## Out of scope for now

`negotiation/`, `body/embodiments/rig_pi/`, `sensorium/fusion/`, and
`cortex/selfmodel/` are specified but not implemented. Pull requests that add
them are welcome in that order; pull requests that add a fourth drive or a
second embodiment before the refusal machine exists are premature.

## Commit messages

One line, active voice, and what changed in the behaviour. "Add viability
filter" is good. "Fix stuff" is not.
