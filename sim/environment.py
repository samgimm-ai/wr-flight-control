"""Atmosphere and gravity model."""

import numpy as np


GRAVITY = np.array([0.0, 0.0, -9.81])  # m/s², NED-style: z points up here → negative
P_ATM = 101325.0  # Pa
RHO_AIR = 1.225   # kg/m³ at sea level


def gravity_vector() -> np.ndarray:
    return GRAVITY.copy()


def atmospheric_pressure(altitude_m: float) -> float:
    """Simple barometric formula."""
    return P_ATM * np.exp(-altitude_m / 8500.0)
