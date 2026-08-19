"""
Guidance: flight phase manager and target generation.

Flight phases:
  LAUNCH      — main engine firing, follow launch angle
  COAST       — engine out, ballistic arc
  FLIP        — attitude reversal to point nozzle downward
  RETRO_BURN  — auxiliary rockets fire to arrest descent
  LANDING     — final vertical descent to target
  LANDED      — mission complete
"""

from __future__ import annotations

from enum import Enum, auto
import numpy as np
from .navigation import NavState


class Phase(Enum):
    LAUNCH = auto()
    COAST = auto()
    FLIP = auto()
    RETRO_BURN = auto()
    LANDING = auto()
    LANDED = auto()


class GuidanceCommand:
    def __init__(
        self,
        phase: Phase,
        target_euler: np.ndarray,       # desired attitude [roll, pitch, yaw] rad
        aux_thrust_fraction: float = 0.0,  # 0–1, aux rocket throttle
        main_thrust_enable: bool = True,
    ):
        self.phase = phase
        self.target_euler = target_euler
        self.aux_thrust_fraction = aux_thrust_fraction
        self.main_thrust_enable = main_thrust_enable


class Guidance:
    """
    State-machine based guidance law.

    Transitions:
      LAUNCH      → COAST      when main engine exhausted
      COAST       → FLIP       when apogee detected (vz ≤ 0) and within glide range
      FLIP        → RETRO_BURN when attitude aligned downward (pitch ≈ -90°)
      RETRO_BURN  → LANDING    when velocity nearly zeroed above target
      LANDING     → LANDED     when altitude ≤ threshold
    """

    FLIP_PITCH_TARGET = -np.pi / 2      # nose down for landing
    LANDED_ALT_M = 0.3                  # m above ground = "landed"
    RETRO_STOP_SPEED_MS = 2.0           # m/s — switch to final descent

    def __init__(self, target_x_m: float, launch_angle_rad: float):
        self.target_x = target_x_m
        self.launch_angle = launch_angle_rad
        self.phase = Phase.LAUNCH

    def update(self, nav: NavState, engine_exhausted: bool) -> GuidanceCommand:
        phase = self.phase

        if phase == Phase.LAUNCH:
            if engine_exhausted:
                self.phase = Phase.COAST
            return GuidanceCommand(
                phase=Phase.LAUNCH,
                target_euler=np.array([0.0, self.launch_angle, 0.0]),
                main_thrust_enable=True,
            )

        if phase == Phase.COAST:
            # Transition to flip at apogee
            if nav.velocity[2] <= 0.0:
                self.phase = Phase.FLIP
            return GuidanceCommand(
                phase=Phase.COAST,
                target_euler=np.array([0.0, self.launch_angle, 0.0]),
                main_thrust_enable=False,
            )

        if phase == Phase.FLIP:
            pitch_err = abs(nav.euler[1] - self.FLIP_PITCH_TARGET)
            if pitch_err < np.radians(10):
                self.phase = Phase.RETRO_BURN
            return GuidanceCommand(
                phase=Phase.FLIP,
                target_euler=np.array([0.0, self.FLIP_PITCH_TARGET, 0.0]),
                main_thrust_enable=False,
            )

        if phase == Phase.RETRO_BURN:
            speed = np.linalg.norm(nav.velocity)
            if speed < self.RETRO_STOP_SPEED_MS and nav.position[2] < 20.0:
                self.phase = Phase.LANDING
            return GuidanceCommand(
                phase=Phase.RETRO_BURN,
                target_euler=np.array([0.0, self.FLIP_PITCH_TARGET, 0.0]),
                aux_thrust_fraction=1.0,
                main_thrust_enable=False,
            )

        if phase == Phase.LANDING:
            if nav.position[2] <= self.LANDED_ALT_M:
                self.phase = Phase.LANDED
            # Gentle descent: partial aux thrust to control fall rate
            return GuidanceCommand(
                phase=Phase.LANDING,
                target_euler=np.array([0.0, self.FLIP_PITCH_TARGET, 0.0]),
                aux_thrust_fraction=0.5,
                main_thrust_enable=False,
            )

        # LANDED
        return GuidanceCommand(
            phase=Phase.LANDED,
            target_euler=np.array([0.0, self.FLIP_PITCH_TARGET, 0.0]),
            aux_thrust_fraction=0.0,
            main_thrust_enable=False,
        )
