"""
WaterRocket Flight Control — simulation entry point.

Usage:
    python main.py                            # single run with scenario defaults
    python main.py --scenario scenarios/baseline.yaml
    python main.py --optimize                 # find optimal launch angle
    python main.py --optimize --angle 38.5    # single run at specific angle
"""

from __future__ import annotations

import argparse
import math
import os
import yaml
import numpy as np

from sim.propulsion import WaterRocketEngine
from sim.aerodynamics import Aerodynamics
from sim.rocket import Rocket
from gnc.navigation import Navigator
from gnc.guidance import Guidance, Phase
from gnc.control import AttitudeController
from viz.plotter import plot_trajectory, plot_range_curve


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


def run_simulation(cfg: dict, angle_deg: float | None = None, verbose: bool = True) -> dict:
    """Run one simulation. Returns result dict."""
    if angle_deg is None:
        angle_deg = cfg["launch"]["angle_deg"]

    angle_rad = math.radians(angle_deg)
    target_x = cfg["target"]["distance_m"]

    rocket = build_rocket(cfg)
    rocket.state.euler[1] = angle_rad

    navigator = Navigator()
    guidance = Guidance(target_x_m=target_x, launch_angle_rad=angle_rad)
    controller = AttitudeController()

    dt = 0.005
    t_max = 20.0

    if verbose:
        print(f"{'='*52}")
        print(f"  Water Rocket Simulation")
        print(f"  Target: {target_x}m  |  Launch angle: {angle_deg:.1f}°")
        print(f"  Initial mass: {rocket.mass():.3f} kg")
        print(f"{'='*52}\n")

    prev_phase = None
    t = 0.0

    while t < t_max:
        s = rocket.state
        nav = navigator.update(s.position, s.velocity, s.euler, s.time)
        guide_cmd = guidance.update(nav, rocket.main_engine.exhausted)

        if verbose and guide_cmd.phase != prev_phase:
            print(f"  t={t:5.2f}s | {guide_cmd.phase.name:12s} | "
                  f"x={s.position[0]:5.1f}m  z={s.position[2]:5.1f}m | "
                  f"v={np.linalg.norm(s.velocity):5.1f}m/s")
            s.phase = guide_cmd.phase.name
            prev_phase = guide_cmd.phase
        else:
            s.phase = guide_cmd.phase.name

        act_cmd = controller.update(s.euler, guide_cmd, dt)
        rocket.step(dt=dt, target_euler=guide_cmd.target_euler, aux_throttle=act_cmd.aux_throttle)
        t += dt

        if guide_cmd.phase == Phase.LANDED:
            break
        if s.position[2] <= 0.0 and t > 0.3:
            break

    final = rocket.state
    miss = np.sqrt((final.position[0] - target_x) ** 2 + final.position[1] ** 2)
    success = miss <= cfg["target"]["radius_m"]

    if verbose:
        print(f"\n{'='*52}")
        print(f"  Landing : x={final.position[0]:.2f}m  y={final.position[1]:.2f}m")
        print(f"  Target  : x={target_x:.2f}m  y=0.00m")
        print(f"  Miss    : {miss:.2f}m  (radius: {cfg['target']['radius_m']}m)")
        print(f"  Speed   : {np.linalg.norm(final.velocity):.2f} m/s")
        print(f"  Time    : {final.time:.2f}s")
        print(f"  Result  : {'✓ SUCCESS' if success else '✗ MISS'}")
        print(f"{'='*52}\n")

    return {
        "landing_x": final.position[0],
        "landing_y": final.position[1],
        "miss_m": miss,
        "success": success,
        "flight_time": final.time,
        "history": rocket.history,
    }


def run_optimize(scenario_path: str):
    from sim.optimizer import optimize

    cfg = load_scenario(scenario_path)
    target_x = cfg["target"]["distance_m"]
    target_r = cfg["target"]["radius_m"]

    print(f"\n{'='*52}")
    print(f"  Launch Angle Optimization")
    print(f"  Target: {target_x}m  radius: {target_r}m")
    print(f"{'='*52}\n")

    result = optimize(cfg, n_grid=37)

    print(f"\n{'='*52}")
    if result["feasible"]:
        print(f"  FEASIBLE")
        print(f"  Optimal angle : {result['optimal_angle']:.2f}°")
        print(f"  Expected range: {result['landing_x']:.2f}m")
        print(f"  Max range     : {result['max_range']:.2f}m @ {result['peak_angle']:.1f}°")
    else:
        deficit = target_x - result["max_range"]
        print(f"  NOT FEASIBLE with current config")
        print(f"  Max achievable: {result['max_range']:.2f}m @ {result['peak_angle']:.1f}°")
        print(f"  Deficit       : {deficit:.2f}m ({deficit/target_x*100:.1f}%)")
        print(f"\n  Suggestions to close the gap:")
        print(f"    - Increase pressure (currently {cfg['main_rocket']['pressure_psi']} PSI)")
        print(f"    - Increase bottle size (currently {cfg['main_rocket']['bottle_volume_L']}L)")
        print(f"    - Reduce dry mass (currently {cfg['main_rocket']['dry_mass_kg']}kg)")
        print(f"    - Reduce Cd / frontal area")
    print(f"{'='*52}\n")

    os.makedirs("data", exist_ok=True)
    plot_range_curve(result, target_x, target_r)

    # Run final simulation at optimal angle with visualization
    print(f"Running simulation at optimal angle ({result['optimal_angle']:.1f}°) …\n")
    sim_result = run_simulation(cfg, angle_deg=result["optimal_angle"], verbose=True)
    plot_trajectory(sim_result["history"], target_distance_m=target_x, target_radius_m=target_r)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Water Rocket Flight Simulator")
    parser.add_argument("--scenario", default="scenarios/baseline.yaml")
    parser.add_argument("--optimize", action="store_true", help="Find optimal launch angle")
    parser.add_argument("--sweep", action="store_true", help="Parameter sweep for minimum specs")
    parser.add_argument("--angle", type=float, default=None, help="Override launch angle (deg)")
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)

    if args.optimize:
        run_optimize(args.scenario)
    elif args.sweep:
        from sim.param_sweep import run_sweep
        cfg = load_scenario(args.scenario)
        run_sweep(cfg)
    else:
        cfg = load_scenario(args.scenario)
        result = run_simulation(cfg, angle_deg=args.angle, verbose=True)
        plot_trajectory(
            result["history"],
            target_distance_m=cfg["target"]["distance_m"],
            target_radius_m=cfg["target"]["radius_m"],
        )
