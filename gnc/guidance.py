"""
Guidance: flight phase manager and target generation.

Flight phases:
  LAUNCH      — main engine firing
  FLIP        — attitude reversal to nose-down (starts at engine cutoff)
  COAST       — nose-down coast; horizontal velocity carries rocket toward target
  RETRO_BURN  — aux rockets fire upward to arrest vertical descent
  LANDING     — final low-speed vertical descent
  LANDED      — mission complete

Key design decisions:
  - Flip starts immediately at engine cutoff (not at apogee) so there is
    enough altitude for the retro-burn phase.
  - Retro-burn fires only when the rocket is descending (vz < threshold),
    so it never opposes ascent.
  - Aux thrust direction is along the nozzle axis (upward when nose-down),
    preserving horizontal velocity.
"""

from __future__ import annotations

from enum import Enum, auto
import numpy as np
from .navigation import NavState


class Phase(Enum):
    LAUNCH = auto()
    FLIP = auto()
    COAST = auto()
    RETRO_BURN = auto()
    LANDING = auto()
    LANDED = auto()


class GuidanceCommand:
    def __init__(
        self,
        phase: Phase,
        target_euler: np.ndarray,
        aux_thrust_fraction: float = 0.0,
        main_thrust_enable: bool = True,
    ):
        self.phase = phase
        self.target_euler = target_euler
        self.aux_thrust_fraction = aux_thrust_fraction
        self.main_thrust_enable = main_thrust_enable


class Guidance:
    """
    State-machine guidance law optimized for maximum horizontal range
    with controlled vertical landing.

    Transitions:
      LAUNCH     → FLIP       at engine cutoff
      FLIP       → COAST      when nose-down within 10° and still ascending (vz > 0)
      COAST      → RETRO_BURN when descending at > RETRO_VZ_THRESHOLD and alt > MIN_RETRO_ALT
      RETRO_BURN → LANDING    when vertical speed < LANDING_VZ_THRESHOLD
      LANDING    → LANDED     when altitude ≤ LANDED_ALT
    """

    NOSE_DOWN = -np.pi / 2              # target pitch for landing attitude

    RETRO_VZ_THRESHOLD = -3.0           # m/s — start retro when falling faster than this
    MIN_RETRO_ALT_M = 4.0               # m   — minimum altitude to begin retro-burn
    LANDING_VZ_THRESHOLD = -2.0         # m/s — transition to slow final descent
    LANDED_ALT_M = 0.3                  # m

    def __init__(self, target_x_m: float, launch_angle_rad: float):
        self.target_x = target_x_m
        self.launch_angle = launch_angle_rad
        self.phase = Phase.LAUNCH

    def update(self, nav: NavState, engine_exhausted: bool) -> GuidanceCommand:
        phase = self.phase

        # ── LAUNCH ──────────────────────────────────────────────────────────
        if phase == Phase.LAUNCH:
            if engine_exhausted:
                self.phase = Phase.FLIP
            return GuidanceCommand(
                phase=Phase.LAUNCH,
                target_euler=np.array([0.0, self.launch_angle, 0.0]),
                main_thrust_enable=True,
            )

        # ── FLIP ─────────────────────────────────────────────────────────────
        # Rotate to nose-down. No thrust — let ballistic arc carry rocket forward.
        if phase == Phase.FLIP:
            pitch_err = abs(nav.euler[1] - self.NOSE_DOWN)
            aligned = pitch_err < np.radians(10)
            if aligned:
                self.phase = Phase.COAST
            return GuidanceCommand(
                phase=Phase.FLIP,
                target_euler=np.array([0.0, self.NOSE_DOWN, 0.0]),
                main_thrust_enable=False,
            )

        # ── COAST ────────────────────────────────────────────────────────────
        # No thrust. Horizontal velocity coasts rocket toward target.
        # Wait for meaningful descent before firing retro.
        if phase == Phase.COAST:
            descending_fast = nav.velocity[2] < self.RETRO_VZ_THRESHOLD
            enough_altitude = nav.position[2] > self.MIN_RETRO_ALT_M
            if descending_fast and enough_altitude:
                self.phase = Phase.RETRO_BURN
            return GuidanceCommand(
                phase=Phase.COAST,
                target_euler=np.array([0.0, self.NOSE_DOWN, 0.0]),
                main_thrust_enable=False,
            )

        # ── RETRO_BURN ───────────────────────────────────────────────────────
        # Aux rockets fire upward (nozzle-axis model) to arrest descent.
        # Throttle is proportional to descent speed — more speed = more thrust.
        if phase == Phase.RETRO_BURN:
            vz = nav.velocity[2]
            if vz > self.LANDING_VZ_THRESHOLD:
                self.phase = Phase.LANDING
            # Proportional throttle: full at -10m/s, tapers off as speed drops
            throttle = float(np.clip(abs(vz) / 10.0, 0.2, 1.0))
            return GuidanceCommand(
                phase=Phase.RETRO_BURN,
                target_euler=np.array([0.0, self.NOSE_DOWN, 0.0]),
                aux_thrust_fraction=throttle,
                main_thrust_enable=False,
            )

        # ── LANDING ──────────────────────────────────────────────────────────
        if phase == Phase.LANDING:
            if nav.position[2] <= self.LANDED_ALT_M:
                self.phase = Phase.LANDED
            return GuidanceCommand(
                phase=Phase.LANDING,
                target_euler=np.array([0.0, self.NOSE_DOWN, 0.0]),
                aux_thrust_fraction=0.3,
                main_thrust_enable=False,
            )

        # ── LANDED ───────────────────────────────────────────────────────────
        return GuidanceCommand(
            phase=Phase.LANDED,
            target_euler=np.array([0.0, self.NOSE_DOWN, 0.0]),
            aux_thrust_fraction=0.0,
            main_thrust_enable=False,
        )
