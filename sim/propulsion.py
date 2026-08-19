"""
Water rocket propulsion model.

Thrust is derived from momentum flux of expelled water,
modeled via isentropic expansion of the trapped air.
"""

from __future__ import annotations

import numpy as np


class WaterRocketEngine:
    """Single water rocket bottle engine."""

    def __init__(
        self,
        bottle_volume_m3: float,
        water_ratio: float,
        initial_pressure_pa: float,
        nozzle_diameter_m: float,
        dry_mass_kg: float,
        water_density_kgm3: float = 1000.0,
    ):
        self.V_total = bottle_volume_m3
        self.V_water0 = bottle_volume_m3 * water_ratio
        self.V_air0 = bottle_volume_m3 * (1 - water_ratio)
        self.P0 = initial_pressure_pa
        self.A_nozzle = np.pi * (nozzle_diameter_m / 2) ** 2
        self.dry_mass = dry_mass_kg
        self.rho_water = water_density_kgm3

        self.water_mass0 = self.V_water0 * water_density_kgm3

        # State
        self.V_water = self.V_water0
        self.exhausted = False

    @property
    def water_mass(self) -> float:
        return self.V_water * self.rho_water

    @property
    def total_mass(self) -> float:
        return self.dry_mass + self.water_mass

    def pressure(self) -> float:
        """Current air pressure (isothermal expansion)."""
        if self.exhausted:
            return 101325.0  # atmospheric
        V_air = self.V_total - self.V_water
        return self.P0 * (self.V_air0 / V_air)

    def thrust_and_mdot(self, P_atm: float = 101325.0) -> tuple[float, float]:
        """
        Returns (thrust_N, mass_flow_rate_kg_s).
        Uses Torricelli-derived exit velocity with gauge pressure.
        """
        if self.exhausted:
            return 0.0, 0.0

        P = self.pressure()
        dP = P - P_atm
        if dP <= 0:
            self.exhausted = True
            return 0.0, 0.0

        v_exit = np.sqrt(2 * dP / self.rho_water)
        mdot = self.rho_water * self.A_nozzle * v_exit  # kg/s
        thrust = mdot * v_exit + dP * self.A_nozzle    # momentum + pressure thrust

        return thrust, mdot

    def step(self, dt: float, P_atm: float = 101325.0) -> tuple[float, float]:
        """Advance one time step. Returns (thrust_N, mdot_kg_s)."""
        thrust, mdot = self.thrust_and_mdot(P_atm)
        if thrust > 0:
            dV = mdot * dt / self.rho_water
            self.V_water = max(0.0, self.V_water - dV)
            if self.V_water == 0.0:
                self.exhausted = True
        return thrust, mdot
