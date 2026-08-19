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
from gnc.navigation import Navigator
from gnc.guidance import Guidance, Phase
from gnc.control import AttitudeController
from viz.plotter import plot_trajectory


def load_scenario(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_rocket(cfg: dict) -> Rocket:
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


def run_simulation(scenario_path: str = "scenarios/baseline.yaml"):
    cfg = load_scenario(scenario_path)
    rocket = build_rocket(cfg)

    launch_angle_deg = cfg["launch"]["angle_deg"]
    launch_angle_rad = math.radians(launch_angle_deg)
    target_x = cfg["target"]["distance_m"]

    rocket.state.euler[1] = launch_angle_rad

    navigator = Navigator()
    guidance = Guidance(target_x_m=target_x, launch_angle_rad=launch_angle_rad)
    controller = AttitudeController()

    dt = 0.005  # 5 ms
    t_max = 15.0

    print(f"{'='*50}")
    print(f"Water Rocket Simulation")
    print(f"Target: {target_x}m | Launch angle: {launch_angle_deg}°")
    print(f"Initial mass: {rocket.mass():.3f} kg")
    print(f"{'='*50}\n")

    prev_phase = None
    t = 0.0

    while t < t_max:
        s = rocket.state

        # Navigation
        nav = navigator.update(s.position, s.velocity, s.euler, s.time)

        # Guidance
        engine_out = rocket.main_engine.exhausted
        guide_cmd = guidance.update(nav, engine_out)

        # Log phase transitions
        if guide_cmd.phase != prev_phase:
            print(f"  t={t:5.2f}s | Phase: {guide_cmd.phase.name:12s} | "
                  f"pos=[{s.position[0]:.1f}, {s.position[2]:.1f}]m | "
                  f"vel={np.linalg.norm(s.velocity):.1f}m/s")
            s.phase = guide_cmd.phase.name
            prev_phase = guide_cmd.phase

        # Control
        act_cmd = controller.update(s.euler, guide_cmd, dt)

        # Dynamics step
        rocket.step(
            dt=dt,
            target_euler=guide_cmd.target_euler,
            aux_throttle=act_cmd.aux_throttle,
        )

        t += dt

        if guide_cmd.phase == Phase.LANDED:
            break
        if s.position[2] <= 0.0 and t > 0.5:
            break

    final = rocket.state
    miss = np.sqrt((final.position[0] - target_x) ** 2 + final.position[1] ** 2)

    print(f"\n{'='*50}")
    print(f"Landing position : x={final.position[0]:.2f}m, y={final.position[1]:.2f}m")
    print(f"Target           : x={target_x:.2f}m, y=0.00m")
    print(f"Miss distance    : {miss:.2f}m  (target radius: {cfg['target']['radius_m']}m)")
    print(f"Final speed      : {np.linalg.norm(final.velocity):.2f} m/s")
    print(f"Flight time      : {final.time:.2f}s")
    success = miss <= cfg["target"]["radius_m"]
    print(f"Result           : {'SUCCESS' if success else 'MISS'}")
    print(f"{'='*50}\n")

    plot_trajectory(
        rocket.history,
        target_distance_m=target_x,
        target_radius_m=cfg["target"]["radius_m"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="scenarios/baseline.yaml")
    args = parser.parse_args()
    run_simulation(args.scenario)
