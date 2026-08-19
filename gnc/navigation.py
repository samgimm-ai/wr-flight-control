"""
Navigation: state estimation from sensor data.

In simulation, ground truth + Gaussian noise is used.
On hardware, this will fuse IMU + barometer + GPS via a Kalman filter.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class NavState:
    position: np.ndarray    # [x, y, z] m
    velocity: np.ndarray    # [vx, vy, vz] m/s
    euler: np.ndarray       # [roll, pitch, yaw] rad
    time: float


class Navigator:
    """
    Dead-reckoning navigator with optional sensor noise injection.
    Wraps ground-truth rocket state for simulation; swapped for
    real sensor fusion on hardware.
    """

    def __init__(self, noise_std: dict | None = None):
        # noise_std keys: position_m, velocity_ms, euler_rad
        self.noise = noise_std or {}

    def update(self, true_position, true_velocity, true_euler, time) -> NavState:
        pos = true_position.copy()
        vel = true_velocity.copy()
        eul = true_euler.copy()

        if self.noise:
            pos += np.random.normal(0, self.noise.get("position_m", 0), 3)
            vel += np.random.normal(0, self.noise.get("velocity_ms", 0), 3)
            eul += np.random.normal(0, self.noise.get("euler_rad", 0), 3)

        return NavState(position=pos, velocity=vel, euler=eul, time=time)
