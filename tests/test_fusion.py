"""Fusion: one vector, fixed channels, documented defaults, monotonic time."""

from __future__ import annotations

from body.embodiments.gridworld import GridWorld
from body.types import Margins, modality_bit
from lab.parity import assert_fusion_parity, assert_fusion_shape_parity
from sensorium.fusion import (
    CHANNELS,
    FusedPerception,
    FusionStage,
    GridWorldBundle,
    PiPerceptionBundle,
    Reading,
    StaticBundle,
    fuse,
)


def healthy() -> Margins:
    return Margins(
        power=0.7,
        thermal=0.6,
        compute=0.5,
        storage=0.4,
        sensor_liveness=1.0,
    )


def full_bundle(t: float = 1.0, body_id: str = "body-0") -> StaticBundle:
    return StaticBundle(
        body_id=body_id,
        t_mono=t,
        t_wall=t,
        seq=int(t),
        data={
            "occupancy": Reading(0.4, 1.0, t),
            "surprise": Reading(0.1, 1.0, t),
            "viability": Reading(0.4, 1.0, t),
            "presence": Reading(1.0, 1.0, t),
            "audio": Reading(0.25, 1.0, t),
        },
    )


def test_grid_bundle_fuses_three_modalities_into_one_vector() -> None:
    body = GridWorld(seed=11)
    bundle = GridWorldBundle.from_snapshot(body.sense(), body.margins())
    fused = fuse(bundle)

    assert fused.channels == CHANNELS
    assert fused.width == 5
    assert 0.0 <= fused.channel("occupancy") <= 1.0
    assert 0.0 <= fused.channel("surprise") <= 1.0
    assert 0.0 <= fused.channel("viability") <= 1.0
    assert fused.channel("audio") == 0.0


def test_missing_modality_uses_the_documented_default() -> None:
    bundle = StaticBundle(
        body_id="body-0",
        t_mono=1.0,
        t_wall=1.0,
        seq=1,
        data={"occupancy": Reading(0.5, 1.0, 1.0)},
    )
    fused = fuse(bundle)

    assert fused.channel("occupancy") == 0.5
    assert fused.channel("audio") == 0.0
    assert fused.freshness[CHANNELS.index("audio")] == 0.0
    assert fused.mask & modality_bit("audio") == 0


def test_timestamp_is_monotonic_even_when_the_bundle_goes_backwards() -> None:
    stage = FusionStage()
    first = stage.fuse(full_bundle(t=5.0))
    second = stage.fuse(full_bundle(t=3.0))

    assert second.t_mono == first.t_mono
    assert second.skew == 2.0


def test_pi_bundle_accepts_camera_pir_and_microphone() -> None:
    bundle = PiPerceptionBundle.from_frames(
        camera=[[0.0, 1.0], [0.5, 0.5]],
        pir=1.0,
        mic_energy=0.25,
        margins=healthy(),
        residual=(0.2, 0.0),
        t_mono=2.0,
        t_wall=2.0,
        seq=2,
    )
    fused = fuse(bundle)

    assert fused.channel("occupancy") == 0.5
    assert fused.channel("presence") == 1.0
    assert fused.channel("audio") == 0.25
    assert fused.channel("surprise") == 0.1


def test_gridworld_and_pi_vectors_have_the_same_shape() -> None:
    body = GridWorld(seed=11)
    grid = fuse(GridWorldBundle.from_snapshot(body.sense(), body.margins()))
    pi = fuse(
        PiPerceptionBundle.from_frames(
            camera=[0.0, 1.0, 0.0, 1.0],
            pir=0.0,
            mic_energy=0.1,
            margins=healthy(),
            residual=body.sense().residual,
        )
    )

    assert_fusion_shape_parity(grid, pi)
    assert grid.channels == pi.channels == CHANNELS


def test_same_bundle_produces_byte_identical_vectors() -> None:
    bundle = full_bundle(t=4.0)
    assert_fusion_parity(fuse(bundle), fuse(bundle))


def test_different_values_fail_strict_fusion_parity() -> None:
    left = fuse(full_bundle(t=4.0))
    right = fuse(full_bundle(t=4.0, body_id="other"))

    try:
        assert_fusion_parity(left, right)
    except AssertionError as exc:
        assert "fusion bytes differ" in str(exc)
    else:
        raise AssertionError("strict fusion parity accepted different bodies")


def test_mask_records_every_present_modality() -> None:
    fused = fuse(full_bundle())

    expected = 0
    for name in CHANNELS:
        expected |= modality_bit(
            {
                "occupancy": "vision",
                "surprise": "dead_reckoning",
                "viability": "thermal",
                "presence": "proximity",
                "audio": "audio",
            }[name]
        )
    assert fused.mask == expected


def test_fusion_stage_state_round_trips() -> None:
    stage = FusionStage()
    stage.fuse(full_bundle(t=1.0))
    restored = FusionStage.from_state(stage.state_dict())
    next_bundle = full_bundle(t=2.0)

    assert stage.fuse(next_bundle).canonical_bytes() == restored.fuse(
        next_bundle
    ).canonical_bytes()


def test_fused_perception_round_trips_through_json_data() -> None:
    fused = fuse(full_bundle(t=6.0))
    restored = FusedPerception.from_dict(fused.to_dict())

    assert restored.canonical_bytes() == fused.canonical_bytes()
