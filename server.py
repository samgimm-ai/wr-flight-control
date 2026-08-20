"""
Water Rocket Simulator — FastAPI web server.

Usage:
    python server.py
    uvicorn server:app --reload --port 8000
"""

from __future__ import annotations

import math
import random
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Water Rocket Simulator")

_HTML_PATH = Path(__file__).parent / "web" / "index.html"


@app.get("/", response_class=HTMLResponse)
async def root():
    return _HTML_PATH.read_text(encoding="utf-8")


# ── Request / Response models ────────────────────────────────────────────────

class SimParams(BaseModel):
    # Environment
    target_distance_m: float = Field(60.0, ge=10, le=200)
    wind_mode: str = Field("manual")           # "manual" | "random"
    wind_speed_ms: float = Field(0.0, ge=0, le=20)
    wind_dir_deg: float = Field(0.0, ge=0, le=360)

    # Rocket hardware
    bottle_volume_L: float = Field(2.0, ge=0.5, le=10.0)
    dry_mass_kg: float = Field(0.30, ge=0.05, le=2.0)
    nozzle_diameter_mm: float = Field(9.0, ge=3.0, le=25.0)
    Cd: float = Field(0.5, ge=0.1, le=1.5)

    # Launch controls (adjustable)
    water_ratio: float = Field(0.33, ge=0.05, le=0.90)
    pressure_psi: float = Field(120.0, ge=20, le=300)
    launch_angle_deg: float = Field(44.0, ge=5, le=85)


class TrajectoryPoint(BaseModel):
    t: float
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    phase: str


# ── Simulation ────────────────────────────────────────────────────────────────

def _run(params: SimParams) -> dict:
    from sim.propulsion import WaterRocketEngine
    from sim.aerodynamics import Aerodynamics
    from sim.rocket import Rocket

    # Resolve wind (random or manual)
    if params.wind_mode == "random":
        wind_speed = random.uniform(0.0, 5.0)
        wind_dir = random.uniform(0.0, 360.0)
    else:
        wind_speed = params.wind_speed_ms
        wind_dir = params.wind_dir_deg

    wind_rad = math.radians(wind_dir)
    wind_v = np.array([
        math.cos(wind_rad) * wind_speed,
        math.sin(wind_rad) * wind_speed,
        0.0,
    ])

    # Rocket body radius estimated from volume (cylindrical, H=3r)
    r_body = (params.bottle_volume_L * 1e-3 / (3 * math.pi)) ** (1 / 3)
    cross_section = math.pi * r_body ** 2

    engine = WaterRocketEngine(
        bottle_volume_m3=params.bottle_volume_L * 1e-3,
        water_ratio=params.water_ratio,
        initial_pressure_pa=params.pressure_psi * 6894.76,
        nozzle_diameter_m=params.nozzle_diameter_mm * 1e-3,
        dry_mass_kg=params.dry_mass_kg,
    )

    aero = Aerodynamics(Cd=params.Cd, cross_section_m2=cross_section)
    rocket = Rocket(main_engine=engine, aux_engine=None, aero=aero)

    angle_rad = math.radians(params.launch_angle_deg)
    rocket.state.euler[1] = angle_rad

    dt = 0.005
    t_max = 20.0
    sample_dt = 0.025          # record every 25 ms
    next_sample = 0.0

    trajectory = []
    max_alt = 0.0
    max_speed = 0.0
    burn_time = 0.0

    t = 0.0
    while t < t_max:
        s = rocket.state

        if t >= next_sample:
            v = s.velocity
            trajectory.append({
                "t": round(t, 3),
                "x": round(float(s.position[0]), 3),
                "y": round(float(s.position[1]), 3),
                "z": round(float(s.position[2]), 3),
                "vx": round(float(v[0]), 3),
                "vy": round(float(v[1]), 3),
                "vz": round(float(v[2]), 3),
                "phase": s.phase,
            })
            next_sample += sample_dt

        max_alt = max(max_alt, float(s.position[2]))
        max_speed = max(max_speed, float(np.linalg.norm(s.velocity)))
        if not engine.exhausted:
            burn_time = t

        # Guidance: maintain launch angle while engine firing, coast after
        target_euler = np.array([0.0, angle_rad, 0.0])
        rocket.step(dt=dt, target_euler=target_euler, wind_velocity=wind_v)
        t += dt

        if s.position[2] <= 0.0 and t > 0.1:
            break

    final = rocket.state
    lx = float(final.position[0])
    ly = float(final.position[1])
    miss = math.sqrt((lx - params.target_distance_m) ** 2 + ly ** 2)

    # Ensure last point is on ground
    if not trajectory or trajectory[-1]["z"] > 0.01:
        trajectory.append({
            "t": round(t, 3),
            "x": round(lx, 3), "y": round(ly, 3), "z": 0.0,
            "vx": 0.0, "vy": 0.0, "vz": 0.0,
            "phase": "LANDED",
        })

    return {
        "trajectory": trajectory,
        "wind": {
            "speed_ms": round(wind_speed, 2),
            "dir_deg": round(wind_dir, 1),
        },
        "stats": {
            "landing_x":   round(lx, 2),
            "landing_y":   round(ly, 2),
            "miss_m":      round(miss, 2),
            "success":     miss <= params.target_distance_m * 0.0167,  # ~1m at 60m
            "flight_time": round(t, 2),
            "max_alt_m":   round(max_alt, 2),
            "max_speed_ms": round(max_speed, 2),
            "burn_time_s": round(burn_time, 3),
            "target_m":    params.target_distance_m,
        },
    }


@app.post("/api/simulate")
async def simulate(params: SimParams):
    try:
        return _run(params)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
