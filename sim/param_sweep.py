"""
Parameter sweep: find minimum hardware specs to reach 60m.

Sweeps over pressure and bottle volume independently,
reports the minimum configuration that hits the target.
"""

from __future__ import annotations

import copy
import numpy as np
import matplotlib.pyplot as plt

from sim.optimizer import simulate_landing_x, optimize


def sweep_pressure(base_cfg: dict, pressures_psi: list[float]) -> list[float]:
    ranges = []
    for psi in pressures_psi:
        cfg = copy.deepcopy(base_cfg)
        cfg["main_rocket"]["pressure_psi"] = psi
        result = optimize(cfg, n_grid=25)
        ranges.append(result["max_range"])
        print(f"  {psi:6.0f} PSI → max range {result['max_range']:.2f}m  "
              f"({'OK' if result['max_range'] >= base_cfg['target']['distance_m'] else 'short'})")
    return ranges


def sweep_volume(base_cfg: dict, volumes_L: list[float]) -> list[float]:
    ranges = []
    for vol in volumes_L:
        cfg = copy.deepcopy(base_cfg)
        cfg["main_rocket"]["bottle_volume_L"] = vol
        result = optimize(cfg, n_grid=25)
        ranges.append(result["max_range"])
        print(f"  {vol:5.2f} L → max range {result['max_range']:.2f}m  "
              f"({'OK' if result['max_range'] >= base_cfg['target']['distance_m'] else 'short'})")
    return ranges


def sweep_dry_mass(base_cfg: dict, masses_kg: list[float]) -> list[float]:
    ranges = []
    for mass in masses_kg:
        cfg = copy.deepcopy(base_cfg)
        cfg["main_rocket"]["dry_mass_kg"] = mass
        result = optimize(cfg, n_grid=25)
        ranges.append(result["max_range"])
        print(f"  {mass:.2f} kg → max range {result['max_range']:.2f}m  "
              f"({'OK' if result['max_range'] >= base_cfg['target']['distance_m'] else 'short'})")
    return ranges


def plot_sweeps(results: dict, target_m: float):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Parameter Sweep — Minimum Specs to Reach Target", fontsize=13, fontweight="bold")

    configs = [
        ("pressure", "Pressure (PSI)", results["pressures"], results["pressure_ranges"]),
        ("volume",   "Bottle Volume (L)", results["volumes"], results["volume_ranges"]),
        ("mass",     "Dry Mass (kg)", results["masses"], results["mass_ranges"]),
    ]

    for ax, (key, xlabel, xs, ys) in zip(axes, configs):
        ax.plot(xs, ys, "b-o", markersize=6, linewidth=2)
        ax.axhline(target_m, color="r", linestyle="--", label=f"Target {target_m}m")
        ax.fill_between(xs, ys, target_m,
                        where=[y >= target_m for y in ys],
                        alpha=0.15, color="green", label="Feasible zone")

        # Mark first feasible point
        for x, y in zip(xs, ys):
            if y >= target_m:
                ax.plot(x, y, "g*", markersize=14)
                ax.annotate(f"{x}", (x, y), textcoords="offset points",
                            xytext=(5, 5), fontsize=9, color="green")
                break

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Max Range (m)", fontsize=11)
        ax.set_title(xlabel.split("(")[0].strip(), fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("data/param_sweep.png", dpi=150)
    print("\n  Saved: data/param_sweep.png")
    plt.close()


def run_sweep(cfg: dict):
    target = cfg["target"]["distance_m"]
    print(f"\n{'='*55}")
    print(f"  Parameter Sweep  (target: {target}m)")
    print(f"  Baseline: {cfg['main_rocket']['pressure_psi']} PSI, "
          f"{cfg['main_rocket']['bottle_volume_L']}L, "
          f"{cfg['main_rocket']['dry_mass_kg']}kg dry")
    print(f"{'='*55}\n")

    pressures = list(range(80, 221, 20))        # 80–220 PSI
    volumes = [round(v, 2) for v in np.arange(1.5, 5.1, 0.5)]  # 1.5–5.0 L
    masses = [round(m, 2) for m in np.arange(0.10, 0.51, 0.05)]  # 0.10–0.50 kg

    print("[ Pressure sweep ]")
    pressure_ranges = sweep_pressure(cfg, pressures)

    print("\n[ Bottle volume sweep ]")
    volume_ranges = sweep_volume(cfg, volumes)

    print("\n[ Dry mass sweep ]")
    mass_ranges = sweep_dry_mass(cfg, masses)

    print(f"\n{'='*55}")
    print("  Minimum specs to reach 60m:")
    for psi, r in zip(pressures, pressure_ranges):
        if r >= target:
            print(f"    Pressure : {psi} PSI  (baseline: {cfg['main_rocket']['pressure_psi']} PSI)")
            break
    for vol, r in zip(volumes, volume_ranges):
        if r >= target:
            print(f"    Volume   : {vol} L   (baseline: {cfg['main_rocket']['bottle_volume_L']} L)")
            break
    for mass, r in zip(masses, mass_ranges):
        if r >= target:
            print(f"    Dry mass : {mass} kg  (baseline: {cfg['main_rocket']['dry_mass_kg']} kg)")
            break
    print(f"{'='*55}\n")

    plot_sweeps({
        "pressures": pressures, "pressure_ranges": pressure_ranges,
        "volumes": volumes, "volume_ranges": volume_ranges,
        "masses": masses, "mass_ranges": mass_ranges,
    }, target)
