"""Aerodynamic drag model (simple axisymmetric body)."""

import numpy as np


class Aerodynamics:
    def __init__(self, Cd: float = 0.5, cross_section_m2: float = 0.003):
        self.Cd = Cd
        self.A = cross_section_m2  # frontal area

    def drag_force(self, velocity_ms: np.ndarray, rho_air: float = 1.225) -> np.ndarray:
        """Returns drag force vector opposing velocity (N)."""
        speed = np.linalg.norm(velocity_ms)
        if speed < 1e-6:
            return np.zeros(3)
        drag_mag = 0.5 * rho_air * self.Cd * self.A * speed ** 2
        return -drag_mag * (velocity_ms / speed)
