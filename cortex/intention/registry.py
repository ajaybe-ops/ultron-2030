"""Name-keyed tests, so a leased intention survives serialisation.

An `Intention` cannot carry closures into SUBSTRATE-1: a lambda has no bytes.
So a test is stored as data — `{"name": ..., "params": {...}}` — and this
registry turns the name back into a callable. Every function here is pure and
takes `(tick, snapshot, margins, params)`. The tick is included because
`lease_expired` cannot be answered without it; the other tests ignore it.

Adding a test means adding a function and one registry entry. Nothing else in
the intention layer needs to know the name.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from body.types import Margins, Snapshot

__all__ = ["REGISTRY", "Test", "evaluate", "make"]

#: A test is pure: same arguments, same answer, no I/O, no mutation.
Test = Callable[[int, Snapshot, Margins, Mapping[str, Any]], bool]


def _mean_abs_residual(snapshot: Snapshot) -> float:
    if not snapshot.residual:
        return 0.0
    return float(sum(abs(r) for r in snapshot.residual) / len(snapshot.residual))


def residual_drop_below(
    tick: int, snapshot: Snapshot, margins: Margins, params: Mapping[str, Any]
) -> bool:
    """True when mean absolute residual has fallen under the threshold."""
    return _mean_abs_residual(snapshot) <= float(params.get("threshold", 0.05))


def margin_recovered(
    tick: int, snapshot: Snapshot, margins: Margins, params: Mapping[str, Any]
) -> bool:
    """True when one named margin has climbed back to the threshold."""
    name = str(params["name"])
    return float(getattr(margins, name)) >= float(params["threshold"])


def lease_expired(
    tick: int, snapshot: Snapshot, margins: Margins, params: Mapping[str, Any]
) -> bool:
    """True once the tick has reached the lease deadline."""
    return int(tick) >= int(params["lease_until"])


def sensor_liveness_restored(
    tick: int, snapshot: Snapshot, margins: Margins, params: Mapping[str, Any]
) -> bool:
    """True when sensor liveness is back above the viability floor."""
    return float(margins.sensor_liveness) >= float(params.get("threshold", 0.9))


def cell_fresh(
    tick: int, snapshot: Snapshot, margins: Margins, params: Mapping[str, Any]
) -> bool:
    """True when one addressed cell has been observed recently enough."""
    try:
        cell = snapshot.cell(str(params.get("frame", "grid")), int(params["slot"]))
    except KeyError:
        return False
    return cell.freshness >= float(params.get("threshold", 0.9))


def cell_var_below(
    tick: int, snapshot: Snapshot, margins: Margins, params: Mapping[str, Any]
) -> bool:
    """True when one addressed cell's predictive variance is small enough."""
    try:
        cell = snapshot.cell(str(params.get("frame", "grid")), int(params["slot"]))
    except KeyError:
        return False
    return cell.var <= float(params.get("threshold", 0.05))


REGISTRY: dict[str, Test] = {
    "residual_drop_below": residual_drop_below,
    "margin_recovered": margin_recovered,
    "lease_expired": lease_expired,
    "sensor_liveness_restored": sensor_liveness_restored,
    "cell_fresh": cell_fresh,
    "cell_var_below": cell_var_below,
}


def make(test_name: str, **params: Any) -> dict[str, Any]:
    """Build a serialisable test spec."""
    if test_name not in REGISTRY:
        raise KeyError(f"unknown test {test_name!r}")
    return {"name": str(test_name), "params": dict(params)}


def evaluate(
    spec: Mapping[str, Any] | None,
    tick: int,
    snapshot: Snapshot,
    margins: Margins,
) -> bool:
    """Run one stored spec. A missing spec never fires."""
    if spec is None:
        return False
    name = str(spec["name"])
    try:
        test = REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown test {name!r} in registry") from exc
    return bool(test(int(tick), snapshot, margins, dict(spec.get("params", {}))))
