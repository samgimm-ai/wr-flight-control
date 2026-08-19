"""Trajectory and optimization visualization."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_trajectory(history, target_distance_m: float = 60.0, target_radius_m: float = 1.0):
    positions = np.array([s.position for s in history])
    times = np.array([s.time for s in history])
    phases = [s.phase for s in history]

    # Color each segment by flight phase
    phase_colors = {
        "LAUNCH": "#e74c3c",
        "COAST": "#3498db",
        "FLIP": "#f39c12",
        "RETRO_BURN": "#9b59b6",
        "LANDING": "#27ae60",
        "LANDED": "#27ae60",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Water Rocket Flight Simulation", fontsize=13, fontweight="bold")

    # Side view (X-Z)
    ax = axes[0]
    current_phase = phases[0]
    seg_start = 0
    for i in range(1, len(phases)):
        if phases[i] != current_phase or i == len(phases) - 1:
            color = phase_colors.get(current_phase, "gray")
            ax.plot(positions[seg_start:i, 0], positions[seg_start:i, 2], color=color, linewidth=2)
            seg_start = i
            current_phase = phases[i]
    ax.axvline(target_distance_m, color="r", linestyle="--", alpha=0.7, label=f"Target {target_distance_m}m")
    ax.set_xlabel("Downrange (m)")
    ax.set_ylabel("Altitude (m)")
    ax.set_title("Trajectory (side view)")
    # Legend for phases
    for phase, color in phase_colors.items():
        ax.plot([], [], color=color, label=phase, linewidth=2)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Altitude vs time
    ax = axes[1]
    ax.plot(times, positions[:, 2], color="#2c3e50", linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Altitude (m)")
    ax.set_title("Altitude vs Time")
    ax.grid(True, alpha=0.3)

    # Landing scatter (top view)
    ax = axes[2]
    target = patches.Circle(
        (target_distance_m, 0), target_radius_m,
        linewidth=2, edgecolor="red", facecolor="none", linestyle="--"
    )
    ax.add_patch(target)
    if len(positions) > 0:
        lx, ly = positions[-1, 0], positions[-1, 1]
        miss = np.sqrt((lx - target_distance_m) ** 2 + ly ** 2)
        ax.plot(lx, ly, "bx", markersize=14, markeredgewidth=2, label=f"Landing ({miss:.1f}m miss)")
    ax.set_xlabel("Downrange (m)")
    ax.set_ylabel("Lateral (m)")
    ax.set_title("Landing accuracy (top view)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    margin = max(5, target_radius_m * 5)
    ax.set_xlim(target_distance_m - margin, target_distance_m + margin)
    ax.set_ylim(-margin, margin)

    plt.tight_layout()
    plt.savefig("data/trajectory.png", dpi=150)
    print("  Saved: data/trajectory.png")
    plt.close()


def plot_range_curve(result: dict, target_x: float, target_radius: float = 1.0):
    angles = result["angles"]
    ranges = result["ranges"]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(angles, ranges, "b-o", markersize=4, linewidth=2, label="Landing range")
    ax.axhline(target_x, color="r", linestyle="--", linewidth=1.5, label=f"Target: {target_x}m")
    ax.axhline(target_x + target_radius, color="r", linestyle=":", alpha=0.5, label=f"±{target_radius}m tolerance")
    ax.axhline(target_x - target_radius, color="r", linestyle=":", alpha=0.5)

    ax.axvline(result["peak_angle"], color="orange", linestyle="--",
               label=f"Peak range angle: {result['peak_angle']:.1f}°")

    if result["feasible"]:
        ax.axvline(result["optimal_angle"], color="green", linestyle="-",
                   label=f"Optimal angle: {result['optimal_angle']:.1f}°")
        ax.plot(result["optimal_angle"], result["landing_x"], "g*", markersize=15)

    ax.set_xlabel("Launch angle (°)", fontsize=12)
    ax.set_ylabel("Landing range (m)", fontsize=12)
    ax.set_title("Launch Angle Optimization — Range vs Angle", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    if not result["feasible"]:
        deficit = target_x - result["max_range"]
        ax.text(
            0.5, 0.05,
            f"MAX RANGE {result['max_range']:.1f}m < TARGET {target_x}m\n"
            f"Deficit: {deficit:.1f}m — need more propellant or less mass",
            transform=ax.transAxes,
            ha="center", fontsize=11, color="red",
            bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="red"),
        )

    plt.tight_layout()
    plt.savefig("data/range_curve.png", dpi=150)
    print("  Saved: data/range_curve.png")
    plt.close()
