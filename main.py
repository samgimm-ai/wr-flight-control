"""
WaterRocket Flight Control — simulation entry point.

Usage:
    python main.py
    python main.py --scenario scenarios/baseline.yaml
"""

import argparse
import math
import yaml
import numpy as np

from sim.propulsion import WaterRocketEngine
from sim.aerodynamics import Aerodynamics
from sim.rocket import Rocket
from viz.plotter import plot_trajectory


def load_scenario(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_rocket(cfg: dict) -> Rocket:
    mc = cfg["main_rocket"]
    engine = WaterRocketEngine(
        bottle_volume_m3=mc["bottle_volume_L"] * 1e-3,
        water_ratio=mc["water_ratio"],
        initial_pressure_pa=mc["pressure_psi"] * 6894.76,
        nozzle_diameter_m=mc["nozzle_diameter_mm"] * 1e-3,
        dry_mass_kg=mc["dry_mass_kg"],
    )
    aero = Aerodynamics(Cd=0.5, cross_section_m2=math.pi * (0.05) ** 2)
    return Rocket(main_engine=engine, aero=aero)


def run_simulation(scenario_path: str = "scenarios/baseline.yaml"):
    cfg = load_scenario(scenario_path)
    rocket = build_rocket(cfg)

    # Initial launch angle
    angle_deg = cfg["launch"]["angle_deg"]
    angle_rad = math.radians(angle_deg)
    rocket.state.euler[1] = angle_rad  # pitch

    dt = 0.005  # 5 ms time step
    t_max = 10.0

    print(f"Simulation start — target: {cfg['target']['distance_m']}m, launch angle: {angle_deg}°")
    print(f"Initial mass: {rocket.mass():.3f} kg")

    t = 0.0
    while t < t_max:
        state = rocket.step(dt)
        t += dt

        # Stop if landed
        if state.position[2] <= 0.0 and t > 0.1:
            break

    final = rocket.state
    print(f"\n--- Results ---")
    print(f"Landing position: x={final.position[0]:.2f}m, y={final.position[1]:.2f}m")
    print(f"Target distance: {cfg['target']['distance_m']}m")
    print(f"Miss distance: {abs(final.position[0] - cfg['target']['distance_m']):.2f}m")
    print(f"Flight time: {final.time:.2f}s")

    plot_trajectory(
        rocket.history,
        target_distance_m=cfg["target"]["distance_m"],
        target_radius_m=cfg["target"]["radius_m"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="scenarios/baseline.yaml")
    args = parser.parse_args()
    run_simulation(args.scenario)
