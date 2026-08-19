"""Trajectory visualization."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def plot_trajectory(history, target_distance_m: float = 60.0, target_radius_m: float = 1.0):
    positions = np.array([s.position for s in history])
    times = np.array([s.time for s in history])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # X-Z (side view)
    axes[0].plot(positions[:, 0], positions[:, 2])
    axes[0].axvline(target_distance_m, color="r", linestyle="--", label="Target")
    axes[0].set_xlabel("Downrange (m)")
    axes[0].set_ylabel("Altitude (m)")
    axes[0].set_title("Trajectory (side view)")
    axes[0].legend()
    axes[0].grid(True)

    # Altitude vs time
    axes[1].plot(times, positions[:, 2])
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Altitude (m)")
    axes[1].set_title("Altitude vs Time")
    axes[1].grid(True)

    # Landing scatter (X-Y top view)
    theta = np.linspace(0, 2 * np.pi, 100)
    axes[2].plot(
        target_distance_m + target_radius_m * np.cos(theta),
        target_radius_m * np.sin(theta),
        "r--", label="Target"
    )
    if len(positions) > 0:
        axes[2].plot(positions[-1, 0], positions[-1, 1], "bx", markersize=12, label="Landing")
    axes[2].set_xlabel("Downrange (m)")
    axes[2].set_ylabel("Lateral (m)")
    axes[2].set_title("Landing accuracy (top view)")
    axes[2].legend()
    axes[2].grid(True)
    axes[2].set_aspect("equal")

    plt.tight_layout()
    plt.savefig("data/trajectory.png", dpi=150)
    plt.show()
