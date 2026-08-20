"""
6-DOF rocket state and dynamics integrator.

Coordinate system: x = downrange, y = lateral, z = altitude (up positive).
Attitude represented as Euler angles [roll, pitch, yaw].
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from .propulsion import WaterRocketEngine
from .aerodynamics import Aerodynamics
from .environment import gravity_vector, atmospheric_pressure, RHO_AIR


@dataclass
class RocketState:
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))   # m  [x, y, z]
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))   # m/s
    euler: np.ndarray = field(default_factory=lambda: np.zeros(3))      # rad [roll, pitch, yaw]
    omega: np.ndarray = field(default_factory=lambda: np.zeros(3))      # rad/s body rates
    time: float = 0.0
    phase: str = "LAUNCH"

    def copy(self) -> "RocketState":
        return RocketState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            euler=self.euler.copy(),
            omega=self.omega.copy(),
            time=self.time,
            phase=self.phase,
        )


class Rocket:
    """
    Point-mass rocket with attitude dynamics.
    Supports main engine + auxiliary thruster system.
    """

    # Attitude slew rate limit (rad/s) — physically constrained by aerodynamics/cold-gas
    MAX_SLEW_RATE = np.radians(180)  # 180°/s (cold-gas thrusters)

    def __init__(
        self,
        main_engine: WaterRocketEngine,
        aux_engine: WaterRocketEngine | None,
        aero: Aerodynamics,
        dry_mass_kg: float = 0.0,
        moment_of_inertia: float = 0.05,  # kg·m²
    ):
        self.main_engine = main_engine
        self.aux_engine = aux_engine        # None if no aux system
        self.aero = aero
        self.extra_dry_mass = dry_mass_kg
        self.Iy = moment_of_inertia

        self.state = RocketState()
        self.history: list[RocketState] = []

    def mass(self) -> float:
        m = self.main_engine.total_mass + self.extra_dry_mass
        if self.aux_engine:
            m += self.aux_engine.total_mass
        return m

    def thrust_direction(self) -> np.ndarray:
        """Unit vector along rocket body axis (thrust direction)."""
        pitch = self.state.euler[1]
        yaw = self.state.euler[2]
        return np.array([
            np.cos(pitch) * np.cos(yaw),
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch),
        ])

    def step(
        self,
        dt: float,
        target_euler: np.ndarray | None = None,
        aux_throttle: float = 0.0,
        wind_velocity: np.ndarray | None = None,
    ) -> RocketState:
        """
        Advance simulation by dt seconds.

        target_euler: desired attitude — applied as a rate-limited slew.
        aux_throttle: 0–1 throttle for auxiliary retro rockets.
        """
        s = self.state
        P_atm = atmospheric_pressure(s.position[2])

        # --- Attitude update (rate-limited slew toward target) ---
        if target_euler is not None:
            euler_error = target_euler - s.euler
            max_delta = self.MAX_SLEW_RATE * dt
            delta = np.clip(euler_error, -max_delta, max_delta)
            s.euler += delta

        # --- Main engine thrust ---
        thrust_main, _ = self.main_engine.step(dt, P_atm)
        t_dir = self.thrust_direction()
        F_thrust = thrust_main * t_dir

        # --- Auxiliary retro thrust (fires out the nozzle = opposite body axis) ---
        # When nose-down (pitch = -90°): nozzle points up → thrust is upward.
        # This brakes vertical descent without killing horizontal velocity.
        F_aux = np.zeros(3)
        if self.aux_engine and aux_throttle > 0.0:
            thrust_aux, _ = self.aux_engine.step(dt * aux_throttle, P_atm)
            nozzle_dir = -self.thrust_direction()   # thrust direction = opposite body axis
            F_aux = thrust_aux * aux_throttle * nozzle_dir

        # --- Forces ---
        F_drag = self.aero.drag_force(s.velocity, RHO_AIR, wind_velocity)
        F_gravity = gravity_vector() * self.mass()
        F_total = F_thrust + F_aux + F_drag + F_gravity

        # --- Euler integration ---
        accel = F_total / self.mass()
        s.velocity += accel * dt
        s.position += s.velocity * dt

        # Ground collision guard
        if s.position[2] < 0.0:
            s.position[2] = 0.0
            s.velocity = np.zeros(3)

        s.time += dt
        self.history.append(s.copy())
        return s.copy()
