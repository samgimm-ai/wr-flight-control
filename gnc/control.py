"""
Control: attitude PID + thrust allocation.

Converts guidance target attitude into actuator commands:
  - Gimbal angle (main nozzle deflection)
  - Auxiliary thruster on/off + throttle
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from .guidance import GuidanceCommand


@dataclass
class ActuatorCommand:
    gimbal_pitch_rad: float = 0.0      # nozzle deflection (main engine)
    gimbal_yaw_rad: float = 0.0
    aux_throttle: float = 0.0          # 0–1 (all aux rockets, equal thrust)


class PIDController:
    def __init__(self, kp: float, ki: float, kd: float, limit: float = np.pi / 6):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.limit = limit
        self._integral = 0.0
        self._prev_error = 0.0

    def update(self, error: float, dt: float) -> float:
        self._integral += error * dt
        derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
        self._prev_error = error

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        return float(np.clip(output, -self.limit, self.limit))

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0


class AttitudeController:
    """
    Dual-axis (pitch + yaw) PID attitude controller.
    Outputs gimbal deflection angles and aux throttle.
    """

    def __init__(self):
        self.pitch_pid = PIDController(kp=2.0, ki=0.1, kd=0.5)
        self.yaw_pid = PIDController(kp=2.0, ki=0.1, kd=0.5)

    def update(
        self,
        current_euler: np.ndarray,
        guidance_cmd: GuidanceCommand,
        dt: float,
    ) -> ActuatorCommand:
        pitch_err = guidance_cmd.target_euler[1] - current_euler[1]
        yaw_err = guidance_cmd.target_euler[2] - current_euler[2]

        gimbal_pitch = self.pitch_pid.update(pitch_err, dt)
        gimbal_yaw = self.yaw_pid.update(yaw_err, dt)

        return ActuatorCommand(
            gimbal_pitch_rad=gimbal_pitch,
            gimbal_yaw_rad=gimbal_yaw,
            aux_throttle=guidance_cmd.aux_thrust_fraction,
        )
