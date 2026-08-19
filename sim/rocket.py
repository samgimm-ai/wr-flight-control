"""
6-DOF rocket state and dynamics integrator.

Coordinate system: x = downrange, y = lateral, z = altitude (up positive).
Attitude represented as Euler angles [roll, pitch, yaw] for simplicity.
"""

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

    def copy(self) -> "RocketState":
        return RocketState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            euler=self.euler.copy(),
            omega=self.omega.copy(),
            time=self.time,
        )


class Rocket:
    """
    Single-stage rocket with optional auxiliary thrusters.
    Simplified: treats rocket as point mass with attitude for thrust direction.
    """

    def __init__(
        self,
        main_engine: WaterRocketEngine,
        aero: Aerodynamics,
        moment_of_inertia: float = 0.05,  # kg·m² (rough estimate)
    ):
        self.main_engine = main_engine
        self.aero = aero
        self.Iy = moment_of_inertia  # pitch/yaw inertia

        self.state = RocketState()
        self.history: list[RocketState] = []

    def mass(self) -> float:
        return self.main_engine.total_mass

    def thrust_direction(self) -> np.ndarray:
        """Unit vector in direction of thrust (body +z axis in world frame)."""
        pitch = self.state.euler[1]
        yaw = self.state.euler[2]
        return np.array([
            np.cos(pitch) * np.cos(yaw),
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch),
        ])

    def step(self, dt: float, gimbal_angle: float = 0.0) -> RocketState:
        """
        Advance simulation by dt seconds.
        gimbal_angle: nozzle deflection angle (rad), positive = pitch up correction.
        """
        s = self.state
        P_atm = atmospheric_pressure(s.position[2])
        thrust, _ = self.main_engine.step(dt, P_atm)

        t_dir = self.thrust_direction()
        F_thrust = thrust * t_dir
        F_drag = self.aero.drag_force(s.velocity, RHO_AIR)
        F_gravity = gravity_vector() * self.mass()

        F_total = F_thrust + F_drag + F_gravity
        accel = F_total / self.mass()

        # Simple Euler integration (will upgrade to RK4 later)
        s.velocity += accel * dt
        s.position += s.velocity * dt

        # Ground collision
        if s.position[2] < 0.0:
            s.position[2] = 0.0
            s.velocity = np.zeros(3)

        s.time += dt
        self.history.append(s.copy())
        return s.copy()
