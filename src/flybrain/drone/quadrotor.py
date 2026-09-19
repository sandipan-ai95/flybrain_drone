"""Advanced quadrotor physics.

A more faithful 6-DoF rigid-body quadrotor model than :class:`SimplePhysicsSim`.
Kept intentionally small (no external deps beyond numpy) so it runs alongside
the neural simulator at real time in the web dashboard.

State conventions
-----------------
* World frame: X-forward, Y-left, Z-up (right-handed).
* Body frame:  x-forward, y-left, z-up.
* Euler angles: intrinsic ZYX (yaw ψ, pitch θ, roll φ).
* Rotors laid out in an ``X`` configuration::

        (0) front-right  CCW+
        (1) front-left   CW-
        (2) rear-left    CCW+
        (3) rear-right   CW-

  Each rotor produces thrust ``T_i = kT · ω_i²`` along +z_body and reaction
  torque ``Q_i = ± kQ · ω_i²`` about z_body (sign = spin direction).

Control mapping
---------------
The public interface still consumes :class:`DroneCommand` with the four
normalized channels (thrust / roll / pitch / yaw ∈ ~[-1, 1]) so nothing
upstream has to change.  Internally we convert those to target rotor
angular velocities via the standard mixing matrix and let the rigid-body
dynamics do the rest.  This means the neural motor decoder still drives
the drone with intuitive commands, but we get:

* real inertia + gyroscopic-like coupling between thrust and tilt,
* separate translational + rotational drag,
* individual rotor spin-up dynamics (first-order lag),
* a mild ground-effect thrust bonus below ~2·R,
* body-frame velocity/accel available for logging.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .physics import DroneState, DroneCommand, clamp_command


# ---------------------------------------------------------------- parameters
@dataclass
class QuadrotorParams:
    mass: float = 0.9                 # kg
    arm: float = 0.20                 # m, rotor-to-CoM distance
    Ixx: float = 6.5e-3               # kg·m²
    Iyy: float = 6.5e-3
    Izz: float = 1.2e-2
    Ir:  float = 4.0e-5               # rotor spin inertia (each)
    kT:  float = 3.2e-5               # thrust  coeff  (N   / (rad/s)²)
    kQ:  float = 6.0e-7               # torque  coeff  (N·m / (rad/s)²)
    w_hover: float = 0.0              # filled in __post_init__
    w_max: float = 900.0              # rad/s, cap
    rotor_tau: float = 0.025          # s, first-order rotor spin-up (snappy)
    drag_lin: float = 0.35            # N / (m/s)
    drag_ang: float = 0.06            # N·m / (rad/s)
    gravity: float = 9.81
    ground_effect_R: float = 0.35     # rotor radius used for GE model
    ground_effect_gain: float = 0.15  # up to +15 % thrust near ground

    def __post_init__(self) -> None:
        # ω that produces T·4 = m·g at hover
        self.w_hover = math.sqrt(self.mass * self.gravity / (4.0 * self.kT))


# ---------------------------------------------------------------- helpers
def _euler_zyx_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll),  math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw),   math.sin(yaw)
    return np.array([
        [cy * cp,  cy * sp * sr - sy * cr,  cy * sp * cr + sy * sr],
        [sy * cp,  sy * sp * sr + cy * cr,  sy * sp * cr - cy * sr],
        [   -sp,             cp * sr,                 cp * cr    ],
    ])


def _body_rates_to_euler_dot(roll: float, pitch: float,
                             wx: float, wy: float, wz: float) -> tuple[float, float, float]:
    """Convert body angular velocity → Euler-angle rates (ZYX)."""
    cr, sr = math.cos(roll),  math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cp = cp if abs(cp) > 1e-4 else (1e-4 if cp >= 0 else -1e-4)
    tp = sp / cp
    dphi   = wx + (sr * tp) * wy + (cr * tp) * wz
    dtheta =            cr * wy   -      sr * wz
    dpsi   =         (sr / cp) * wy + (cr / cp) * wz
    return dphi, dtheta, dpsi


# ---------------------------------------------------------------- sim
class QuadrotorPhysicsSim:
    """6-DoF rigid-body quadrotor with per-rotor dynamics.

    Public interface matches :class:`SimplePhysicsSim` so it can be swapped
    without touching the rest of the pipeline.
    """

    def __init__(self, initial: DroneState | None = None,
                 params: QuadrotorParams | None = None) -> None:
        self.p = params or QuadrotorParams()
        self._initial = initial or DroneState()
        self.state = DroneState(**vars(self._initial))
        # rotor angular velocities (rad/s) — start at hover so we don't fall on tick 0
        self.rotor_w = np.full(4, self.p.w_hover, dtype=np.float64)
        self.rotor_w_ref = self.rotor_w.copy()
        self._bounds = ((-25.0, 25.0), (-25.0, 25.0), (0.0, 12.0))
        # signs: CCW+, CW-, CCW+, CW-
        self._spin = np.array([+1.0, -1.0, +1.0, -1.0])

    # ------------------------------------------------------------------ ctrl
    def reset(self) -> None:
        self.state = DroneState(**vars(self._initial))
        self.rotor_w[:] = self.p.w_hover
        self.rotor_w_ref[:] = self.p.w_hover

    def _mix(self, cmd: DroneCommand) -> np.ndarray:
        """Command → target rotor ω via linearised mix around hover.

        cmd.thrust ~ [-1,1] : net vertical accel above gravity
        cmd.roll   ~ [-1,1] : body-x torque (right wing down = +)
        cmd.pitch  ~ [-1,1] : body-y torque (nose up      = +)
        cmd.yaw    ~ [-1,1] : body-z torque (CCW           = +)
        """
        wh = self.p.w_hover
        # scale thrust: cmd.thrust=+1 → roughly +40 % rotor speed
        base = wh * (1.0 + 0.40 * float(cmd.thrust))
        droll  = 90.0 * float(cmd.roll)
        dpitch = 90.0 * float(cmd.pitch)
        dyaw   = 45.0 * float(cmd.yaw)
        # X-config mixing
        w = np.array([
            base - droll + dpitch - dyaw,   # front-right, CCW
            base + droll + dpitch + dyaw,   # front-left,  CW
            base + droll - dpitch - dyaw,   # rear-left,   CCW
            base - droll - dpitch + dyaw,   # rear-right,  CW
        ], dtype=np.float64)
        return np.clip(w, 0.0, self.p.w_max)

    # ------------------------------------------------------------------ step
    def step(self, cmd: DroneCommand, dt: float) -> DroneState:
        cmd = clamp_command(cmd)
        p = self.p
        s = self.state

        # 1) rotor first-order lag toward mix target
        self.rotor_w_ref = self._mix(cmd)
        alpha = 1.0 - math.exp(-dt / max(p.rotor_tau, 1e-3))
        self.rotor_w += alpha * (self.rotor_w_ref - self.rotor_w)

        # 2) rotor forces / torques in the body frame
        w2 = self.rotor_w ** 2
        T = p.kT * w2                                  # per-rotor thrust
        # Ground effect (very simple): T *= 1 + gain·(R/z)² clipped to a small bonus
        if s.z < 4.0 * p.ground_effect_R and s.z > 0.02:
            ge = 1.0 + p.ground_effect_gain * (p.ground_effect_R / max(s.z, 0.05)) ** 2
            T = T * min(ge, 1.0 + p.ground_effect_gain)
        Ttot = float(T.sum())

        a = p.arm / math.sqrt(2.0)   # X-config lever arm
        tau_x = a * ( (T[0] + T[3]) - (T[1] + T[2]) ) * -1.0     # roll
        tau_y = a * ( (T[0] + T[1]) - (T[2] + T[3]) )            # pitch
        tau_z = float((self._spin * p.kQ * w2).sum())            # yaw (reaction)

        # 3) rigid-body: body → world thrust
        R = _euler_zyx_to_R(s.roll, s.pitch, s.yaw)
        thrust_world = R @ np.array([0.0, 0.0, Ttot])
        # gravity + drag
        v = np.array([s.vx, s.vy, s.vz])
        f_drag = -p.drag_lin * v * np.linalg.norm(v) ** 0.0   # linear-ish
        f = thrust_world + np.array([0.0, 0.0, -p.mass * p.gravity]) + f_drag
        a_world = f / p.mass

        # 4) rotational dynamics in body frame
        wx, wy, wz = s.wx, s.wy, s.wz
        Ix, Iy, Iz = p.Ixx, p.Iyy, p.Izz
        dwx = (tau_x - (Iz - Iy) * wy * wz - p.drag_ang * wx) / Ix
        dwy = (tau_y - (Ix - Iz) * wx * wz - p.drag_ang * wy) / Iy
        dwz = (tau_z - (Iy - Ix) * wx * wy - p.drag_ang * wz) / Iz

        # 5) integrate (semi-implicit Euler)
        s.vx += a_world[0] * dt
        s.vy += a_world[1] * dt
        s.vz += a_world[2] * dt
        s.x  += s.vx * dt
        s.y  += s.vy * dt
        s.z  += s.vz * dt

        s.wx += dwx * dt
        s.wy += dwy * dt
        s.wz += dwz * dt
        dphi, dtheta, dpsi = _body_rates_to_euler_dot(
            s.roll, s.pitch, s.wx, s.wy, s.wz)
        s.roll  += dphi   * dt
        s.pitch += dtheta * dt
        s.yaw   += dpsi   * dt
        # Keep pitch in (-π/2 + eps, π/2 - eps) to avoid gimbal singularity
        lim = math.pi / 2 - 0.05
        s.pitch = max(-lim, min(lim, s.pitch))

        # 6) world bounds
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = self._bounds
        s.collided = False
        if s.z <= zmin:
            s.z = zmin
            s.vz = max(s.vz, 0.0)
            # damp bounce
            s.wx *= 0.5; s.wy *= 0.5
            s.collided = True
        if not (xmin <= s.x <= xmax) or not (ymin <= s.y <= ymax) or s.z >= zmax:
            s.collided = True

        s.t += dt
        return s

    # ------------------------------------------------------------------ extra
    def rotor_speeds(self) -> np.ndarray:
        """Current rotor angular velocities (rad/s), shape (4,)."""
        return self.rotor_w.copy()

    def rotor_speeds_norm(self) -> np.ndarray:
        """Rotor ω / ω_max, useful for visualization."""
        return self.rotor_w / self.p.w_max
