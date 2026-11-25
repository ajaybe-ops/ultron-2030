"""Concrete embodiments. The simulated gridworld is implemented; the Pi rig
lives in `body/embodiments/rig_pi/` in a later build step and must satisfy the
same port without touching the core."""

from body.embodiments.gridworld import GridWorld

__all__ = ["GridWorld"]
