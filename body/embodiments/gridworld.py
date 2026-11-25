"""Simulated 10x10 embodiment: the body that runs on a laptop with no hardware.

The gridworld exists so the agent core can be developed and replayed without a
Raspberry Pi, while obeying the same port contract as the rig. It has one
agent, one passive-proximity sensor (the PIR stand-in), and a 100-cell visual
field that is only fresh within `SENSE_RADIUS` of the agent. Everything the
core sees is produced by `sense()`; nothing else is exposed.

Noise is a pure function of `(seed, tick, slot)`. There is no RNG object, so
there is no hidden state to serialise and replay stays byte-identical.

Margins are all "headroom, higher is better": moving burns power and thermal
headroom; resting recovers both; sensor liveness is 1.0 unless a sensor is
explicitly disabled, in which case the corresponding cells decay instead of
disappearing.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from body.port import Body
from body.types import (
    Affordance,
    ActionResult,
    Cell,
    Margins,
    Snapshot,
    modality_bit,
)

__all__ = ["GridWorld", "GRID_H", "GRID_W", "PROXIMITY_SLOT", "SLOT_COUNT"]

GRID_W = 10
GRID_H = 10
GRID_SLOTS = GRID_W * GRID_H
PROXIMITY_SLOT = GRID_SLOTS
SLOT_COUNT = GRID_SLOTS + 1
SENSE_RADIUS = 3
STATE_KIND = "gridworld-1"

_MASK = (1 << 64) - 1
_VISION = modality_bit("vision")
_PROXIMITY = modality_bit("proximity")

#: Starting cell state before anything has ever been observed: maximally
#: ignorant, not absent.
_PRIOR_VALUE = 0.5
_PRIOR_VAR = 0.25
_PRIOR_FRESHNESS = 0.0

_NOISE_AMPLITUDE = 0.015
_UNOBSERVED_DECAY = 0.82
_UNOBSERVED_VAR_GROWTH = 1.12
_UNOBSERVED_VAR_BIAS = 0.004
_VAR_CEILING = 0.25


def _mix(seed: int, *parts: int) -> int:
    """Deterministic 64-bit mix; the stand-in for a seeded RNG."""
    h = (1469598103934665603 ^ (int(seed) & _MASK)) & _MASK
    for part in parts:
        h ^= int(part) & _MASK
        h = (h * 1099511628211) & _MASK
        h ^= h >> 29
        h &= _MASK
    return h


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class GridWorld(Body):
    """A deterministic 10x10 body with vision, proximity, and finite margins.

    Construction is pure; `load_state()` restores an exact mid-session body,
    which is what replay uses to re-execute a recorded tick.
    """

    def __init__(
        self,
        seed: int = 0,
        body_id: str = "gridworld-0",
        t_wall0: float = 1_700_000_000.0,
        dt: float = 0.5,
    ) -> None:
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.seed = int(seed)
        self.body_id = body_id
        self.t_wall0 = float(t_wall0)
        self.dt = float(dt)

        self.tick = 0
        self.x = 0
        self.y = 0
        self.proximity_enabled = True
        self.last_action = "INIT"

        self._margins = Margins(
            power=1.0,
            thermal=1.0,
            compute=0.9,
            storage=0.95,
            sensor_liveness=1.0,
        )
        self._last: list[tuple[float, float, float, float]] = [
            (_PRIOR_VALUE, _PRIOR_VAR, 0.0, _PRIOR_FRESHNESS)
            for _ in range(SLOT_COUNT)
        ]
        self._forced: dict[int, int] = {}
        self._cache: Snapshot | None = None
        self._cache_tick = -1

        self._world: tuple[float, ...] = tuple(
            self._truth_for(slot) for slot in range(GRID_SLOTS)
        )

    # ------------------------------------------------------------- world model

    def _truth_for(self, slot: int) -> float:
        """Ground truth occupancy for a cell; never read by the core directly."""
        if slot == 0:
            return 0.0
        return 1.0 if _mix(self.seed, slot, 17) % 100 < 14 else 0.0

    def _noise(self, slot: int) -> float:
        raw = _mix(self.seed, self.tick, slot, 31) % 2001
        return (raw / 1000.0 - 1.0) * _NOISE_AMPLITUDE

    def _proximity_value(self) -> float:
        nearest = min(
            self._chebyshev(self.x, self.y, slot % GRID_W, slot // GRID_W)
            for slot in range(GRID_SLOTS)
            if self._world[slot] > 0.5
        ) if any(v > 0.5 for v in self._world) else GRID_W
        span = max(GRID_W - 1, 1)
        return _clamp(1.0 - nearest / span)

    @staticmethod
    def _chebyshev(ax: int, ay: int, bx: int, by: int) -> int:
        return max(abs(ax - bx), abs(ay - by))

    # ------------------------------------------------------------------- clock

    @property
    def t_mono(self) -> float:
        return self.tick * self.dt

    @property
    def t_wall(self) -> float:
        return self.t_wall0 + self.t_mono

    # -------------------------------------------------------------------- port

    def sense(self) -> Snapshot:
        """Fuse vision and proximity into one snapshot with a real residual.

        Cells within the sensing radius are refreshed; everything else carries
        its previous value with decaying freshness and rising variance. The
        residual is `|observed - previous|` per cell — a naive one-step
        persistence predictor, computed here so no other layer has to.
        """
        if self._cache is not None and self._cache_tick == self.tick:
            return self._cache

        now = self.t_mono
        cells: list[Cell] = []
        residual: list[float] = []
        updated: list[tuple[float, float, float, float]] = []

        for slot in range(SLOT_COUNT):
            previous = self._last[slot]
            if slot == PROXIMITY_SLOT:
                observed = self.proximity_enabled
                distance = 0
                truth = self._proximity_value()
                mask = _PROXIMITY
            else:
                distance = self._chebyshev(
                    self.x, self.y, slot % GRID_W, slot // GRID_W
                )
                forced = self._forced.get(slot)
                observed = distance <= SENSE_RADIUS or (
                    forced is not None and forced == self.tick
                )
                truth = self._world[slot]
                mask = _VISION

            if observed:
                value = _clamp(truth + self._noise(slot))
                if distance == 0:
                    var = 0.0004
                elif slot == PROXIMITY_SLOT:
                    var = 0.0009
                else:
                    var = 0.0004 + 0.0016 * (distance / SENSE_RADIUS)
                freshness = 1.0
                observed_at = now
                residual_value = abs(value - previous[0])
            else:
                value = previous[0]
                var = min(_VAR_CEILING, previous[1] * _UNOBSERVED_VAR_GROWTH + _UNOBSERVED_VAR_BIAS)
                freshness = previous[3] * _UNOBSERVED_DECAY
                observed_at = previous[2]
                residual_value = 0.0

            cells.append(
                Cell(
                    frame="grid" if slot != PROXIMITY_SLOT else "proximity",
                    slot=slot,
                    value=value,
                    var=var,
                    observed_at=observed_at,
                    freshness=freshness,
                    modality_mask=mask,
                )
            )
            residual.append(residual_value)
            updated.append((value, var, observed_at, freshness))

        self._last = updated
        snapshot = Snapshot(
            t_mono=now,
            t_wall=self.t_wall,
            seq=self.tick,
            body_id=self.body_id,
            cells=tuple(cells),
            residual=tuple(residual),
        )
        self._cache = snapshot
        self._cache_tick = self.tick
        return snapshot

    def actuate(self, action: Mapping[str, Any]) -> ActionResult:
        """Perform MOVE, OBSERVE, or REST; report cost and outcome."""
        verb = str(action.get("verb", "")).upper().strip()
        params = {k: v for k, v in action.items() if k != "verb"}
        before = self._margins.as_dict()

        if verb == "MOVE":
            ok, detail = self._do_move(action)
        elif verb == "OBSERVE":
            ok, detail = self._do_observe(action)
        elif verb == "REST":
            ok, detail = self._do_rest()
        else:
            return ActionResult(
                ok=False,
                verb=verb or "UNKNOWN",
                params=params,
                t_mono=self.t_mono,
                detail="UNKNOWN_VERB",
                margin_delta={},
            )

        after = self._margins.as_dict()
        delta = {name: after[name] - before[name] for name in after}
        return ActionResult(
            ok=ok,
            verb=verb,
            params=params,
            t_mono=self.t_mono,
            detail=detail,
            margin_delta=delta,
        )

    def declare_affordances(self) -> Sequence[Affordance]:
        """The verbs this body genuinely supports, with their nominal costs."""
        return (
            Affordance(
                verb="MOVE",
                params=("dx", "dy"),
                cost={"power": 0.012, "thermal": 0.018},
                description="Step up to one cell in x and y; blocked cells refuse.",
            ),
            Affordance(
                verb="OBSERVE",
                params=("slot",),
                cost={"compute": 0.002, "power": 0.001},
                description="Force a high-confidence reading of one grid slot.",
            ),
            Affordance(
                verb="REST",
                params=(),
                cost={"compute": 0.0},
                description="Hold position; recover power and thermal headroom.",
            ),
        )

    def margins(self) -> Margins:
        return self._margins

    # --------------------------------------------------------------- actuators

    def _do_move(self, action: Mapping[str, Any]) -> tuple[bool, str]:
        try:
            dx = int(action.get("dx", 0))
            dy = int(action.get("dy", 0))
        except (TypeError, ValueError):
            return False, "BAD_PARAMS"
        if not (-1 <= dx <= 1 and -1 <= dy <= 1):
            return False, "OUT_OF_RANGE"
        if dx == 0 and dy == 0:
            return False, "NO_OP"

        self._advance()
        steps = max(abs(dx), abs(dy))
        target_x = min(GRID_W - 1, max(0, self.x + dx))
        target_y = min(GRID_H - 1, max(0, self.y + dy))
        blocked = self._world[target_y * GRID_W + target_x] > 0.5

        if blocked:
            self._burn(power=0.004 * steps, thermal=0.006 * steps)
            self.last_action = "MOVE_BLOCKED"
            return True, "BLOCKED"

        self.x = target_x
        self.y = target_y
        self._burn(power=0.012 * steps, thermal=0.018 * steps, compute=0.0005)
        self.last_action = "MOVE"
        return True, f"MOVED:{self.x},{self.y}"

    def _do_observe(self, action: Mapping[str, Any]) -> tuple[bool, str]:
        try:
            slot = int(action.get("slot", -1))
        except (TypeError, ValueError):
            return False, "BAD_PARAMS"
        if not (0 <= slot < GRID_SLOTS):
            return False, "OUT_OF_RANGE"

        self._advance()
        self._forced[slot] = self.tick
        self._burn(power=0.001, thermal=0.002, compute=0.002)
        self.last_action = "OBSERVE"
        return True, f"OBSERVED:{slot}"

    def _do_rest(self) -> tuple[bool, str]:
        self._advance()
        self._recover(power=0.05, thermal=0.05, compute=0.01)
        self.last_action = "REST"
        return True, "RESTED"

    # ----------------------------------------------------------------- helpers

    def _advance(self) -> None:
        self.tick += 1
        self._cache = None
        self._cache_tick = -1

    def _burn(
        self,
        power: float = 0.0,
        thermal: float = 0.0,
        compute: float = 0.0,
        storage: float = 0.0,
    ) -> None:
        m = self._margins
        self._margins = Margins(
            power=_clamp(m.power - power),
            thermal=_clamp(m.thermal - thermal),
            compute=_clamp(m.compute - compute),
            storage=_clamp(m.storage - max(storage, 0.0002)),
            sensor_liveness=self._sensor_liveness(),
        )

    def _recover(
        self, power: float = 0.0, thermal: float = 0.0, compute: float = 0.0
    ) -> None:
        m = self._margins
        self._margins = Margins(
            power=_clamp(m.power + power),
            thermal=_clamp(m.thermal + thermal),
            compute=_clamp(m.compute + compute),
            storage=m.storage,
            sensor_liveness=self._sensor_liveness(),
        )

    def _sensor_liveness(self) -> float:
        return 1.0 if self.proximity_enabled else 0.0

    # -------------------------------------------------------- body-local extras

    def disable_sensor(self, name: str = "proximity") -> None:
        """Kill a sensor on purpose; liveness drops and its cells go stale."""
        if name != "proximity":
            raise ValueError(f"unknown sensor {name!r}")
        self.proximity_enabled = False
        self._margins = Margins(
            power=self._margins.power,
            thermal=self._margins.thermal,
            compute=self._margins.compute,
            storage=self._margins.storage,
            sensor_liveness=0.0,
        )
        self._cache = None

    def enable_sensor(self, name: str = "proximity") -> None:
        if name != "proximity":
            raise ValueError(f"unknown sensor {name!r}")
        self.proximity_enabled = True
        self._margins = Margins(
            power=self._margins.power,
            thermal=self._margins.thermal,
            compute=self._margins.compute,
            storage=self._margins.storage,
            sensor_liveness=1.0,
        )
        self._cache = None

    def state_dict(self) -> dict[str, Any]:
        """Exact body state for replay reconstruction (not part of the port)."""
        return {
            "kind": STATE_KIND,
            "seed": self.seed,
            "body_id": self.body_id,
            "t_wall0": self.t_wall0,
            "dt": self.dt,
            "tick": self.tick,
            "x": self.x,
            "y": self.y,
            "proximity_enabled": self.proximity_enabled,
            "last_action": self.last_action,
            "margins": self._margins.as_dict(),
            "last": [[v, var, at, fresh] for (v, var, at, fresh) in self._last],
            "forced": {str(k): int(v) for k, v in self._forced.items()},
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "GridWorld":
        if state.get("kind") != STATE_KIND:
            raise ValueError(f"not a {STATE_KIND} state: {state.get('kind')!r}")
        body = cls(
            seed=int(state["seed"]),
            body_id=str(state["body_id"]),
            t_wall0=float(state["t_wall0"]),
            dt=float(state["dt"]),
        )
        body.tick = int(state["tick"])
        body.x = int(state["x"])
        body.y = int(state["y"])
        body.proximity_enabled = bool(state["proximity_enabled"])
        body.last_action = str(state["last_action"])
        body._margins = Margins.from_dict(state["margins"])
        body._last = [
            (float(v), float(var), float(at), float(fresh))
            for (v, var, at, fresh) in state["last"]
        ]
        body._forced = {int(k): int(v) for k, v in state["forced"].items()}
        body._cache = None
        body._cache_tick = -1
        return body
