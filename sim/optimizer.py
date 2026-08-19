"""
Launch angle optimizer.

Strategy:
  1. Grid search over angles → range vs angle curve
  2. Find peak (maximum achievable range)
  3. If target is reachable, bisect to find precise angle
  4. Report feasibility
"""

from __future__ import annotations

import math
import numpy as np
from scipy.optimize import brentq

from sim.propulsion import WaterRocketEngine
from sim.aerodynamics import Aerodynamics
from sim.rocket import Rocket
from gnc.navigation import Navigator
from gnc.guidance import Guidance, Phase
from gnc.control import AttitudeController


def build_rocket_from_cfg(cfg: dict) -> Rocket:
    mc = cfg["main_rocket"]
    main_engine = WaterRocketEngine(
        bottle_volume_m3=mc["bottle_volume_L"] * 1e-3,
        water_ratio=mc["water_ratio"],
        initial_pressure_pa=mc["pressure_psi"] * 6894.76,
        nozzle_diameter_m=mc["nozzle_diameter_mm"] * 1e-3,
        dry_mass_kg=mc["dry_mass_kg"],
    )
    ac = cfg.get("aux_rocket")
    aux_engine = None
    if ac:
        aux_engine = WaterRocketEngine(
            bottle_volume_m3=ac["bottle_volume_L"] * ac["count"] * 1e-3,
            water_ratio=ac["water_ratio"],
            initial_pressure_pa=ac["pressure_psi"] * 6894.76,
            nozzle_diameter_m=ac["nozzle_diameter_mm"] * 1e-3,
            dry_mass_kg=0.1 * ac["count"],
        )
    aero = Aerodynamics(Cd=0.5, cross_section_m2=math.pi * (0.05) ** 2)
    return Rocket(main_engine=main_engine, aux_engine=aux_engine, aero=aero)


def simulate_landing_x(angle_deg: float, cfg: dict, dt: float = 0.005) -> float:
    """Run one simulation, return landing x (m)."""
    rocket = build_rocket_from_cfg(cfg)
    angle_rad = math.radians(angle_deg)
    rocket.state.euler[1] = angle_rad

    target_x = cfg["target"]["distance_m"]
    navigator = Navigator()
    guidance = Guidance(target_x_m=target_x, launch_angle_rad=angle_rad)
    controller = AttitudeController()

    t = 0.0
    t_max = 20.0

    while t < t_max:
        s = rocket.state
        nav = navigator.update(s.position, s.velocity, s.euler, s.time)
        guide_cmd = guidance.update(nav, rocket.main_engine.exhausted)
        act_cmd = controller.update(s.euler, guide_cmd, dt)

        rocket.step(dt=dt, target_euler=guide_cmd.target_euler, aux_throttle=act_cmd.aux_throttle)
        t += dt

        if guide_cmd.phase == Phase.LANDED:
            break
        if s.position[2] <= 0.0 and t > 0.3:
            break

    return float(rocket.state.position[0])


def range_error(angle_deg: float, cfg: dict) -> float:
    """Signed error: landing_x - target_x. Used for root finding."""
    return simulate_landing_x(angle_deg, cfg) - cfg["target"]["distance_m"]


def optimize(cfg: dict, n_grid: int = 37) -> dict:
    """
    Returns:
      feasible       : bool — can target be reached at all?
      optimal_angle  : float — best launch angle (deg)
      landing_x      : float — expected landing x (m)
      max_range      : float — maximum achievable range (m)
      peak_angle     : float — angle that gives max range (deg)
      angles         : list[float] — grid angles
      ranges         : list[float] — corresponding ranges
    """
    target_x = cfg["target"]["distance_m"]

    # --- Grid search ---
    angles = list(np.linspace(15.0, 85.0, n_grid))
    print(f"  Grid search: {len(angles)} angles ({angles[0]:.0f}°–{angles[-1]:.0f}°) …")
    ranges = []
    for i, a in enumerate(angles):
        r = simulate_landing_x(a, cfg)
        ranges.append(r)
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(angles)} — {a:.1f}° → {r:.1f}m")

    max_range = max(ranges)
    peak_idx = int(np.argmax(ranges))
    peak_angle = angles[peak_idx]

    print(f"\n  Peak range: {max_range:.2f}m @ {peak_angle:.1f}°  (target: {target_x}m)")

    if max_range < target_x:
        return {
            "feasible": False,
            "optimal_angle": peak_angle,
            "landing_x": max_range,
            "max_range": max_range,
            "peak_angle": peak_angle,
            "angles": angles,
            "ranges": ranges,
        }

    # --- Bisection to find precise angle ---
    # Range curve is unimodal: find left side where range < target (small angle)
    # and right side where range < target (large angle).
    # The root exists on each side of the peak.
    # Prefer the shallower angle (left side) for a flatter trajectory.
    left_angle, right_angle = None, None
    for i in range(peak_idx):
        if ranges[i] < target_x <= ranges[i + 1]:
            left_angle = (angles[i], angles[i + 1])
            break

    if left_angle is not None:
        lo, hi = left_angle
        optimal = brentq(range_error, lo, hi, args=(cfg,), xtol=0.1)
        landing_x = simulate_landing_x(optimal, cfg)
    else:
        optimal = peak_angle
        landing_x = max_range

    return {
        "feasible": True,
        "optimal_angle": optimal,
        "landing_x": landing_x,
        "max_range": max_range,
        "peak_angle": peak_angle,
        "angles": angles,
        "ranges": ranges,
    }
