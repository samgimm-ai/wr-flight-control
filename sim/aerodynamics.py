"""Aerodynamic drag model (simple axisymmetric body)."""

from __future__ import annotations
import numpy as np


class Aerodynamics:
    def __init__(self, Cd: float = 0.5, cross_section_m2: float = 0.003):
        self.Cd = Cd
        self.A = cross_section_m2

    def drag_force(
        self,
        velocity_ms: np.ndarray,
        rho_air: float = 1.225,
        wind_velocity: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Returns drag force vector (N).
        Drag is based on velocity relative to the air (wind-corrected).
        """
        v_rel = velocity_ms - (wind_velocity if wind_velocity is not None else np.zeros(3))
        speed = np.linalg.norm(v_rel)
        if speed < 1e-6:
            return np.zeros(3)
        drag_mag = 0.5 * rho_air * self.Cd * self.A * speed ** 2
        return -drag_mag * (v_rel / speed)
