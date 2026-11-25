"""D1 — the epistemic drive. Wants prediction error to fall without looking away.

Urgency (dossier §4.1):

    ŝ_E = clamp(w_r·r̂ + w_h·Ĥ − w_c·Ĉ)        w_r=0.5, w_h=0.3, w_c=0.2

* `r̂`  — mean residual over reachable cells, divided by a running 95th
  percentile kept in `DriveState.residual_ring`, clamped to [0, 1]. The ring is
  the only memory this drive has, and `urgency()` is the only place it grows.
* `Ĥ`  — mean predictive variance over frontier cells, normalised by
  `VAR_REFERENCE` and clamped. Frontier = `freshness < 0.3`.
* `Ĉ`  — coverage: the fraction of reachable cells that are currently fresh.
  The dossier words this as "observed within the last 40 ticks"; the snapshot
  carries no tick duration, so freshness (`>= 0.3`) is the equivalent
  window. This is a documented simplification, not a hidden one.

Reachability is approximated by modality membership: a cell the body can sense
is a cell it can act on. The port deliberately exposes no geometry, so a drive
cannot know more than that.

The `−Ĉ` term is the anti-rumination brake: staring at a corner is not
curiosity.
"""

from __future__ import annotations

from typing import Sequence

from body.types import Margins, Snapshot, modality_bit
from cortex.drives.state import DriveState
from cortex.goal import Goal

__all__ = ["URGENCY_WEIGHTS", "VAR_REFERENCE", "FRONTIER_THRESHOLD", "candidates", "urgency"]

URGENCY_WEIGHTS = {"residual": 0.5, "frontier": 0.3, "coverage": 0.2}
VAR_REFERENCE = 0.25
FRONTIER_THRESHOLD = 0.3
OBSERVE_COST = {"power": 0.001, "thermal": 0.002, "compute": 0.002}
PROBE_COST = {"power": 0.002, "thermal": 0.003, "compute": 0.004}

_VISION = modality_bit("vision")


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _reachable(snapshot: Snapshot) -> list[int]:
    return [
        index
        for index, cell in enumerate(snapshot.cells)
        if cell.modality_mask & _VISION
    ]


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return float(ordered[index])


def residual_norm(snapshot: Snapshot, reachable: Sequence[int]) -> float:
    """Mean absolute residual over reachable cells."""
    if not reachable:
        return 0.0
    return float(sum(abs(snapshot.residual[i]) for i in reachable) / len(reachable))


def urgency(
    snapshot: Snapshot,
    margins: Margins,
    ledger: object,
    state: DriveState,
) -> float:
    """Compute D1 urgency and grow the residual ring by one sample."""
    reachable = _reachable(snapshot)
    mean_residual = residual_norm(snapshot, reachable)
    state.push_residual(mean_residual)

    p95 = _percentile(state.residual_ring, 0.95)
    r_hat = _clamp(mean_residual / max(p95, 1e-6))

    frontier = [i for i in reachable if snapshot.cells[i].freshness < FRONTIER_THRESHOLD]
    if frontier:
        h_hat = _clamp(
            sum(snapshot.cells[i].var for i in frontier)
            / (len(frontier) * VAR_REFERENCE)
        )
    else:
        h_hat = 0.0

    if reachable:
        fresh = sum(1 for i in reachable if snapshot.cells[i].freshness >= FRONTIER_THRESHOLD)
        c_hat = fresh / len(reachable)
    else:
        c_hat = 0.0

    weights = URGENCY_WEIGHTS
    return _clamp(
        weights["residual"] * r_hat
        + weights["frontier"] * h_hat
        - weights["coverage"] * c_hat
    )


def candidates(
    snapshot: Snapshot,
    margins: Margins,
    ledger: object,
    state: DriveState,
) -> list[Goal]:
    """Emit OBSERVE and PROBE candidates, ranked by information gain per cost.

    PROBE is emitted only for cells that are fresh but still uncertain; the
    gridworld does not declare PROBE, so the affordance filter removes these
    before scoring. That is the filter doing its job, not dead code.
    """
    reachable = _reachable(snapshot)
    out: list[Goal] = []

    for index in reachable:
        cell = snapshot.cells[index]
        if cell.freshness < FRONTIER_THRESHOLD:
            gain = _clamp(cell.var / VAR_REFERENCE)
            if gain <= 0.0:
                continue
            out.append(
                Goal(
                    verb="OBSERVE",
                    drive="D1",
                    value=gain,
                    cost_vector=dict(OBSERVE_COST),
                    reversibility=1.0,
                    latency=1,
                    expected_information_gain=gain,
                    target=(cell.frame, cell.slot),
                )
            )
        elif cell.var > 0.0005:
            gain = _clamp(cell.var / VAR_REFERENCE) * 0.5
            out.append(
                Goal(
                    verb="PROBE",
                    drive="D1",
                    value=gain,
                    cost_vector=dict(PROBE_COST),
                    reversibility=0.8,
                    latency=2,
                    expected_information_gain=gain,
                    target=(cell.frame, cell.slot),
                )
            )

    out.sort(
        key=lambda g: (
            -(g.expected_information_gain / (g.cost() + 1e-9)),
            g.verb,
            g.target or ("", -1),
        )
    )
    return out
