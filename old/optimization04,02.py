# -*- coding: utf-8 -*-
"""
optimize_required_only.py

Minimal required coaxial propeller optimization + report plots (single script).

Implements:
- Fixed Ru/Rl = 1 (NOT optimized)
- Optimize variables: independent chord (upper+lower, 4 ctrl pts each), independent linear twist (upper+lower),
  and tip speed via omega.
- 30 radial stations, root cutout = 0.1R
- Objective: minimize shaft power at design flight state at target thrust (per coax unit)
- Sweep: climb speed Vc +/- 10 m/s in 1 m/s steps, trimming omega (RPM) to match target thrust at each speed
- Writes at end: results NPZ + summary TXT + YAML upper/lower + PNG plots
- Uses relative paths only

Dependencies expected next to this file:
- fluid.py
- propeller.py
- bemt_coaxial.py
- access_clcd.py (used by BEMT for airfoil polars)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import copy
import os
import numpy as np
import matplotlib.pyplot as plt

from functools import partial
from scipy.optimize import minimize, Bounds, differential_evolution, brentq
from scipy.interpolate import PchipInterpolator

import fluid
import propeller
import bemt_coaxial


# =============================================================================
# Config
# =============================================================================
@dataclass(frozen=True)
class Config:
    # Aircraft & design state
    mass_kg: float = 650.0
    g: float = 9.81
    altitude_m: float = 500.0
    V_design: float = 3.0

    # Rotor system
    n_propulsors: int = 8          # number of coaxial units on aircraft
    disk_loading: float = 160.0    # N/m^2 for the coaxial unit (total disk area of both rotors)
    n_blades: int = 2

    # Limits
    M_tip_max: float = 0.40

    # Geometry discretization
    n_stations: int = 30
    r_root_norm: float = 0.10

    # Dimensional chord reference (the nondimensional chord is c/c_ref)
    c_ref_override_m: float = 0.15

    # Airfoil
    force_airfoil_id: int | None = 2412

    # Optimization settings
    de_maxiter: int = 20
    de_popsize: int = 10
    stage2_maxiter: int = 200

    stage1_coupling_iters: int = 5
    stage2_coupling_iters: int = 10
    strict_coupling_iters: int = 25

    # Penalty weights (DE stage)
    W_THRUST_SHORTFALL: float = 6000.0
    W_MACH: float = 2500.0
    W_SHAPE: float = 1500.0
    W_COVER_R: float = 12000.0

    # Hard constraints (COBYLA)
    MAX_C_OVER_R: float = 0.25
    THRUST_TOL_N: float = 5.0

    # Input file (relative)
    yaml_path: str = os.path.join("data", "pybemt_tmotor28.yaml")

    # Output (relative)
    results_root: str = "results"
    out_yaml_base: str = os.path.join("data", "pybemt_optimized_coax")  # writes *_upper.yaml, *_lower.yaml
    checkpoint_file: str = "checkpoint_required_only.npy"

    # Sweep / trim
    sweep_delta: int = 10
    sweep_step: int = 1
    trim_coupling_iters: int = 15
    trim_bracket_expand: float = 1.6
    trim_max_expand: int = 12


# =============================================================================
# Helpers
# =============================================================================
class EvalCache:
    """Cache to avoid duplicate expensive evaluations during COBYLA constraints."""
    def __init__(self, atol: float = 1e-9):
        self.atol = float(atol)
        self.x_last = None
        self.result = None

    def get(self, x, *, base_rot1, base_rot2, fl, V_inf, r_R, cfg: Config, iters: int, W: float):
        x = np.asarray(x, dtype=float)
        if self.x_last is not None and np.allclose(x, self.x_last, rtol=0.0, atol=self.atol):
            return self.result
        self.result = evaluate_system(
            x,
            base_rot1=base_rot1,
            base_rot2=base_rot2,
            fl=fl,
            V_inf=float(V_inf),
            r_R=r_R,
            cfg=cfg,
            max_coupling_iter=int(iters),
            W=float(W),
        )
        self.x_last = x.copy()
        return self.result


def radial_grid(n_stations: int, r_root_norm: float) -> np.ndarray:
    return np.linspace(float(r_root_norm), 1.0, int(n_stations))


def compute_equal_radii_from_DL(*, W: float, disk_loading: float, n_propulsors: int) -> tuple[float, float]:
    """
    Fixed Ru/Rl=1 sizing. Total coaxial unit area A_total = (W/N_propulsors)/DL.
    A_total = 2*pi*R^2 -> R = sqrt(A_total/(2*pi))
    """
    A_total = (float(W) / float(n_propulsors)) / float(disk_loading)
    R = float(np.sqrt(A_total / (2.0 * np.pi)))
    return R, R


def linear_pitch(r_R: np.ndarray, beta_root_deg: float, beta_tip_deg: float) -> np.ndarray:
    r0 = float(r_R[0])
    r1 = float(r_R[-1])
    t = (r_R - r0) / max(r1 - r0, 1e-12)
    return float(beta_root_deg) + (float(beta_tip_deg) - float(beta_root_deg)) * t


def chord_pchip(r_R: np.ndarray, c_ctrl: np.ndarray) -> np.ndarray:
    """
    4 chord control points at [r0, 0.45, 0.75, 1.0] -> c/c_ref distribution via PCHIP.
    """
    r_pts = np.array([float(r_R[0]), 0.45, 0.75, 1.0], dtype=float)
    c_pts = np.array(c_ctrl, dtype=float)
    interp = PchipInterpolator(r_pts, c_pts)
    c = interp(r_R)
    return np.maximum(c, 1e-4)


def override_scale_in_memory(rot: propeller.Propeller, *, R_new: float, c_ref_new: float, nblades_new: int):
    rot.radius = float(R_new)
    rot.c_ref = float(c_ref_new)
    rot.nblades = int(nblades_new)
    return rot


def shape_violations_chord(c_ctrl: np.ndarray) -> tuple[float, float, float]:
    """Preferred chord trend: peak near 0.45R and decreasing toward tip; root not larger than peak."""
    c_root, c_045, c_075, c_tip = map(float, c_ctrl)
    v1 = max(0.0, c_075 - c_045)   # enforce c045 >= c075
    v2 = max(0.0, c_tip - c_075)   # enforce c075 >= ctip
    v3 = max(0.0, c_root - c_045)  # enforce c045 >= croot
    return v1, v2, v3


def shape_violations_twist(beta_root: float, beta_tip: float) -> float:
    """Washout: beta_root >= beta_tip."""
    return max(0.0, float(beta_tip) - float(beta_root))


def shape_penalty(x: np.ndarray, P: float, cfg: Config) -> float:
    x = np.asarray(x, dtype=float)
    cU = x[0:4]
    cL = x[4:8]
    bU_root, bU_tip = x[8], x[9]
    bL_root, bL_tip = x[10], x[11]
    v = []
    v.extend(shape_violations_chord(cU))
    v.extend(shape_violations_chord(cL))
    v.append(shape_violations_twist(bU_root, bU_tip))
    v.append(shape_violations_twist(bL_root, bL_tip))
    return float(cfg.W_SHAPE * sum(vi * vi for vi in v) * max(float(P), 1.0))


# =============================================================================
# Build rotors from design vector
# =============================================================================
def build_rotors_from_x(x, *, base_rot1, base_rot2, r_R, cfg: Config, W: float):
    """
    Design variables (13):
      x[0:4]   upper chord control points
      x[4:8]   lower chord control points
      x[8:10]  upper twist root/tip (deg)
      x[10:12] lower twist root/tip (deg)
      x[12]    omega (rad/s)
    """
    x = np.asarray(x, dtype=float)

    cU_ctrl = x[0:4]
    cL_ctrl = x[4:8]
    bU_root, bU_tip = float(x[8]), float(x[9])
    bL_root, bL_tip = float(x[10]), float(x[11])
    omega = float(x[12])

    Ru, Rl = compute_equal_radii_from_DL(W=W, disk_loading=cfg.disk_loading, n_propulsors=cfg.n_propulsors)

    chordU = chord_pchip(r_R, cU_ctrl)
    chordL = chord_pchip(r_R, cL_ctrl)
    pitchU = linear_pitch(r_R, bU_root, bU_tip)
    pitchL = linear_pitch(r_R, bL_root, bL_tip)

    r1 = copy.deepcopy(base_rot1)
    r2 = copy.deepcopy(base_rot2)

    r1 = override_scale_in_memory(r1, R_new=Ru, c_ref_new=cfg.c_ref_override_m, nblades_new=cfg.n_blades)
    r2 = override_scale_in_memory(r2, R_new=Rl, c_ref_new=cfg.c_ref_override_m, nblades_new=cfg.n_blades)

    r1.r_R = np.array(r_R, dtype=float)
    r2.r_R = np.array(r_R, dtype=float)
    r1.chord = np.array(chordU, dtype=float)
    r2.chord = np.array(chordL, dtype=float)
    r1.pitch = np.array(pitchU, dtype=float)
    r2.pitch = np.array(pitchL, dtype=float)

    if cfg.force_airfoil_id is not None:
        r1.airfoil = np.full_like(r1.r_R, int(cfg.force_airfoil_id), dtype=int)
        r2.airfoil = np.full_like(r2.r_R, int(cfg.force_airfoil_id), dtype=int)

    r1.omega = omega
    r2.omega = omega
    return r1, r2


# =============================================================================
# Physics evaluation
# =============================================================================
def _compute_lower_inflow_from_upper(out1: dict, *, V_inf: float, r_R: np.ndarray, R1: float, R2: float) -> np.ndarray:
    """
    Reconstruct lower-rotor inflow using the simplified wake model:
    contraction factor = 1/sqrt(2), far wake velocity ~ 2*vi.
    """
    r1_phys = r_R * float(R1)
    r2_phys = r_R * float(R2)
    contraction = 1.0 / np.sqrt(2.0)

    vi1 = np.asarray(out1["Vax"], dtype=float) - float(V_inf)
    wake_far = 2.0 * vi1
    r_wake = r1_phys * contraction
    wake_on_R2 = np.interp(r2_phys, r_wake, wake_far, left=0.0, right=0.0)
    return float(V_inf) + wake_on_R2


def evaluate_system(x, *, base_rot1, base_rot2, fl, V_inf: float, r_R, cfg: Config, max_coupling_iter: int, W: float):
    r1, r2 = build_rotors_from_x(x, base_rot1=base_rot1, base_rot2=base_rot2, r_R=r_R, cfg=cfg, W=W)

    totals, out1, out2, coupling = bemt_coaxial.coaxial_bemt_fixed(
        r1, r2, fl,
        V_inf=float(V_inf),
        omega1=float(r1.omega),
        omega2=float(r2.omega),
        r_R=r_R,
        max_coupling_iter=int(max_coupling_iter),
    )

    omega = float(np.asarray(x, dtype=float)[12])
    P_shaft = float(totals["Q_total"] * omega)

    c1 = np.max(np.asarray(r1.chord) * float(r1.c_ref))
    c2 = np.max(np.asarray(r2.chord) * float(r2.c_ref))
    max_c_over_R = max(c1 / float(r1.radius), c2 / float(r2.radius))

    Rmax = max(float(r1.radius), float(r2.radius))
    M_tip = float(omega) * Rmax / float(fl.a)

    V2_inflow = _compute_lower_inflow_from_upper(out1, V_inf=float(V_inf), r_R=np.asarray(r_R), R1=float(r1.radius), R2=float(r2.radius))
    w1 = np.asarray(out1["Vax"], dtype=float) - float(V_inf)
    w2 = np.asarray(out2["Vax"], dtype=float) - np.asarray(V2_inflow, dtype=float)

    return {
        "totals": totals,
        "out1": out1,
        "out2": out2,
        "P_shaft": P_shaft,
        "r1": r1,
        "r2": r2,
        "max_c_over_R": float(max_c_over_R),
        "M_tip": float(M_tip),
        "V2_inflow": np.asarray(V2_inflow, dtype=float),
        "w1": np.asarray(w1, dtype=float),
        "w2": np.asarray(w2, dtype=float),
    }


# =============================================================================
# Stage 1 objective (DE)
# =============================================================================
def objective_stage1(x, *, base_rot1, base_rot2, fl, V_inf, r_R, cfg: Config, T_target: float, W: float):
    res = evaluate_system(
        x,
        base_rot1=base_rot1,
        base_rot2=base_rot2,
        fl=fl,
        V_inf=float(V_inf),
        r_R=r_R,
        cfg=cfg,
        max_coupling_iter=cfg.stage1_coupling_iters,
        W=W,
    )

    P = float(res["P_shaft"])
    if (not np.isfinite(P)) or P <= 1.0:
        return 1e12

    T = float(res["totals"]["T_total"])

    shortfall = max(0.0, (T_target - T) / max(T_target, 1e-12))
    penalty_T = cfg.W_THRUST_SHORTFALL * (shortfall ** 2) * P

    mach_margin = cfg.M_tip_max - float(res["M_tip"])
    penalty_M = cfg.W_MACH * (max(0.0, -mach_margin) ** 2) * P

    penalty_shape = shape_penalty(x, P, cfg)

    v = max(0.0, float(res["max_c_over_R"]) - cfg.MAX_C_OVER_R)
    penalty_cR = cfg.W_COVER_R * (v ** 2) * P

    return float(P + penalty_T + penalty_M + penalty_shape + penalty_cR)


# =============================================================================
# Stage 2 constraints (COBYLA ineq: fun(x) >= 0)
# =============================================================================
def c_thrust_min(x, *, cache: EvalCache, base_rot1, base_rot2, fl, V_inf, r_R, cfg: Config, T_target: float, W: float):
    res = cache.get(x, base_rot1=base_rot1, base_rot2=base_rot2, fl=fl, V_inf=V_inf, r_R=r_R, cfg=cfg, iters=cfg.stage2_coupling_iters, W=W)
    return float(res["totals"]["T_total"] - T_target)


def c_thrust_max(x, *, cache: EvalCache, base_rot1, base_rot2, fl, V_inf, r_R, cfg: Config, T_target: float, W: float):
    res = cache.get(x, base_rot1=base_rot1, base_rot2=base_rot2, fl=fl, V_inf=V_inf, r_R=r_R, cfg=cfg, iters=cfg.stage2_coupling_iters, W=W)
    return float((T_target + cfg.THRUST_TOL_N) - res["totals"]["T_total"])


def c_mach(x, *, base_rot1, base_rot2, fl, V_inf, r_R, cfg: Config, W: float):
    res = evaluate_system(x, base_rot1=base_rot1, base_rot2=base_rot2, fl=fl, V_inf=V_inf, r_R=r_R, cfg=cfg, max_coupling_iter=1, W=W)
    return float(cfg.M_tip_max - res["M_tip"])


def c_max_coverR(x, *, base_rot1, base_rot2, fl, V_inf, r_R, cfg: Config, W: float):
    res = evaluate_system(x, base_rot1=base_rot1, base_rot2=base_rot2, fl=fl, V_inf=V_inf, r_R=r_R, cfg=cfg, max_coupling_iter=1, W=W)
    return float(cfg.MAX_C_OVER_R - res["max_c_over_R"])


# Analytic shape constraints (help COBYLA avoid weird chord control ordering)
def cU_c045_ge_c075(x): return float(np.asarray(x, float)[1] - np.asarray(x, float)[2])
def cU_c075_ge_ctip(x): return float(np.asarray(x, float)[2] - np.asarray(x, float)[3])
def cU_c045_ge_croot(x): return float(np.asarray(x, float)[1] - np.asarray(x, float)[0])

def cL_c045_ge_c075(x): return float(np.asarray(x, float)[5] - np.asarray(x, float)[6])
def cL_c075_ge_ctip(x): return float(np.asarray(x, float)[6] - np.asarray(x, float)[7])
def cL_c045_ge_croot(x): return float(np.asarray(x, float)[5] - np.asarray(x, float)[4])

def c_washout_upper(x): return float(np.asarray(x, float)[8] - np.asarray(x, float)[9])
def c_washout_lower(x): return float(np.asarray(x, float)[10] - np.asarray(x, float)[11])


# =============================================================================
# Trim / sweep
# =============================================================================
def _thrust_error_for_omega(omega: float, x_geom: np.ndarray, *, V_inf: float, base_rot1, base_rot2, fl, r_R, cfg: Config, W: float, T_target: float):
    x = np.array(x_geom, dtype=float)
    x[12] = float(omega)
    res = evaluate_system(
        x, base_rot1=base_rot1, base_rot2=base_rot2, fl=fl, V_inf=float(V_inf),
        r_R=r_R, cfg=cfg, max_coupling_iter=cfg.trim_coupling_iters, W=W
    )
    return float(res["totals"]["T_total"] - T_target), res


def trim_omega_for_thrust(x_geom: np.ndarray, *, omega_init: float, omega_min: float, omega_max: float,
                          V_inf: float, base_rot1, base_rot2, fl, r_R, cfg: Config, W: float, T_target: float):
    """
    Trim omega to satisfy T_total == T_target using brentq; expand bracket if needed.
    """
    lo = max(float(omega_min), 0.05)
    hi = float(omega_max)

    lo = max(lo, float(omega_init) / cfg.trim_bracket_expand)
    hi = min(hi, float(omega_init) * cfg.trim_bracket_expand)

    f_lo, _ = _thrust_error_for_omega(lo, x_geom, V_inf=V_inf, base_rot1=base_rot1, base_rot2=base_rot2, fl=fl, r_R=r_R, cfg=cfg, W=W, T_target=T_target)
    f_hi, _ = _thrust_error_for_omega(hi, x_geom, V_inf=V_inf, base_rot1=base_rot1, base_rot2=base_rot2, fl=fl, r_R=r_R, cfg=cfg, W=W, T_target=T_target)

    expand = 0
    while f_lo * f_hi > 0.0 and expand < cfg.trim_max_expand:
        lo = max(float(omega_min), lo / cfg.trim_bracket_expand)
        hi = min(float(omega_max), hi * cfg.trim_bracket_expand)
        f_lo, _ = _thrust_error_for_omega(lo, x_geom, V_inf=V_inf, base_rot1=base_rot1, base_rot2=base_rot2, fl=fl, r_R=r_R, cfg=cfg, W=W, T_target=T_target)
        f_hi, _ = _thrust_error_for_omega(hi, x_geom, V_inf=V_inf, base_rot1=base_rot1, base_rot2=base_rot2, fl=fl, r_R=r_R, cfg=cfg, W=W, T_target=T_target)
        expand += 1

    if f_lo * f_hi > 0.0:
        # fallback: pick best endpoint
        omega_pick = lo if abs(f_lo) < abs(f_hi) else hi
        _, res_pick = _thrust_error_for_omega(omega_pick, x_geom, V_inf=V_inf, base_rot1=base_rot1, base_rot2=base_rot2, fl=fl, r_R=r_R, cfg=cfg, W=W, T_target=T_target)
        return float(omega_pick), res_pick

    def f(om):
        val, _ = _thrust_error_for_omega(om, x_geom, V_inf=V_inf, base_rot1=base_rot1, base_rot2=base_rot2, fl=fl, r_R=r_R, cfg=cfg, W=W, T_target=T_target)
        return val

    omega_trim = float(brentq(f, lo, hi, xtol=1e-6, rtol=1e-8, maxiter=80))
    _, res_trim = _thrust_error_for_omega(omega_trim, x_geom, V_inf=V_inf, base_rot1=base_rot1, base_rot2=base_rot2, fl=fl, r_R=r_R, cfg=cfg, W=W, T_target=T_target)
    return omega_trim, res_trim


# =============================================================================
# Plotting
# =============================================================================
def _savefig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def make_plots_from_npz(npz_path: str, out_dir: str, cfg: Config):
    d = np.load(npz_path, allow_pickle=False)

    r_R = d["r_R"]

    # geometry
    chordU = d["chordU"]; chordL = d["chordL"]
    pitchU = d["pitchU"]; pitchL = d["pitchL"]
    radiusU = float(d["radiusU"]); radiusL = float(d["radiusL"])
    c_refU = float(d["c_refU"]); c_refL = float(d["c_refL"])

    # distributions
    phiU = d["phiU"]; phiL = d["phiL"]
    dTdrU = d["dTdrU"]; dTdrL = d["dTdrL"]
    alphaU = d["alphaU"]; alphaL = d["alphaL"]
    wU = d["wU"]; wL = d["wL"]

    # sweep
    Vs = d["Vs"]
    P_shaft_sweep = d["P_shaft_sweep"]
    P_excess_sweep = d["P_excess_sweep"]
    RPM_sweep = d["RPM_sweep"]
    Mtip_sweep = d["Mtip_sweep"]

    # --- Fig: chord
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(r_R, chordU * c_refU / radiusU, label="Upper")
    plt.plot(r_R, chordL * c_refL / radiusL, label="Lower")
    plt.grid(True); plt.xlabel("r/R [-]"); plt.ylabel("c/R [-]")
    plt.title("Chord distribution (upper & lower)")
    plt.legend()
    _savefig(os.path.join(out_dir, "fig_chord.png"))

    # --- Fig: twist
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(r_R, pitchU, label="Upper")
    plt.plot(r_R, pitchL, label="Lower")
    plt.grid(True); plt.xlabel("r/R [-]"); plt.ylabel("Pitch β [deg]")
    plt.title("Twist (pitch) distribution (upper & lower)")
    plt.legend()
    _savefig(os.path.join(out_dir, "fig_twist.png"))

    # --- Fig: inflow phi
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(r_R, np.degrees(phiU), label="Upper")
    plt.plot(r_R, np.degrees(phiL), label="Lower")
    plt.grid(True); plt.xlabel("r/R [-]"); plt.ylabel("Inflow angle φ [deg]")
    plt.title("Inflow angle distribution")
    plt.legend()
    _savefig(os.path.join(out_dir, "fig_phi.png"))

    # --- Fig: thrust loading (normalized)
    plt.figure(figsize=(7.5, 4.5))
    dT1n = dTdrU / max(np.trapz(dTdrU, r_R), 1e-12)
    dT2n = dTdrL / max(np.trapz(dTdrL, r_R), 1e-12)
    plt.plot(r_R, dT1n, label="Upper (norm)")
    plt.plot(r_R, dT2n, label="Lower (norm)")
    plt.grid(True); plt.xlabel("r/R [-]"); plt.ylabel("Normalized (1/T) dT/dr")
    plt.title("Thrust loading distribution")
    plt.legend()
    _savefig(os.path.join(out_dir, "fig_dTdr.png"))

    # --- Fig: AoA (recommended, helps report)
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(r_R, np.degrees(alphaU), label="Upper")
    plt.plot(r_R, np.degrees(alphaL), label="Lower")
    plt.grid(True); plt.xlabel("r/R [-]"); plt.ylabel("Angle of attack α [deg]")
    plt.title("Angle of attack distribution")
    plt.legend()
    _savefig(os.path.join(out_dir, "fig_alpha.png"))

    # --- Fig: induced velocity (recommended, helps report)
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(r_R, wU, label="Upper induced w")
    plt.plot(r_R, wL, label="Lower induced w")
    plt.grid(True); plt.xlabel("r/R [-]"); plt.ylabel("Induced axial velocity w [m/s]")
    plt.title("Induced velocity distribution")
    plt.legend()
    _savefig(os.path.join(out_dir, "fig_induced_w.png"))

    # --- Fig: power vs speed (required)
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(Vs, P_shaft_sweep)
    plt.grid(True); plt.xlabel("Climb speed Vc [m/s]"); plt.ylabel("Shaft power [W]")
    plt.title("Power vs climb speed (trimmed RPM)")
    _savefig(os.path.join(out_dir, "fig_power_vs_speed.png"))

    # --- Fig: power excluding climb power (required)
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(Vs, P_excess_sweep)
    plt.grid(True); plt.xlabel("Climb speed Vc [m/s]"); plt.ylabel("P_shaft - T*Vc [W]")
    plt.title("Power excluding climb power (trimmed RPM)")
    _savefig(os.path.join(out_dir, "fig_power_excluding_climb.png"))

    # --- Fig: trimmed RPM vs speed (recommended)
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(Vs, RPM_sweep)
    plt.grid(True); plt.xlabel("Climb speed Vc [m/s]"); plt.ylabel("RPM (trimmed)")
    plt.title("Trimmed RPM vs climb speed")
    _savefig(os.path.join(out_dir, "fig_rpm_vs_speed.png"))

    # --- Fig: tip Mach vs speed (recommended sanity check)
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(Vs, Mtip_sweep)
    plt.axhline(cfg.M_tip_max, linestyle="--")
    plt.grid(True); plt.xlabel("Climb speed Vc [m/s]"); plt.ylabel("Tip Mach [-]")
    plt.title("Tip Mach over sweep (trimmed)")
    _savefig(os.path.join(out_dir, "fig_mtip_vs_speed.png"))


# =============================================================================
# Output formatting
# =============================================================================
def rotor_to_yaml_dict(rot: propeller.Propeller) -> dict:
    return {
        "propeller": {
            "nblades": int(rot.nblades),
            "radius": float(rot.radius),
            "omega_rpm": float(rot.omega * 30.0 / np.pi),
            "c_ref": float(rot.c_ref),
            "geometry": {
                "r_R": np.asarray(rot.r_R, dtype=float).tolist(),
                "pitch_deg": np.asarray(rot.pitch, dtype=float).tolist(),
                "chord_over_c_ref": np.asarray(rot.chord, dtype=float).tolist(),
                "airfoil": np.asarray(rot.airfoil, dtype=int).tolist(),
            }
        }
    }


def build_summary(cfg: Config, final: dict, x_opt: np.ndarray, T_target: float, W: float, stage2_result) -> str:
    r1 = final["r1"]
    r2 = final["r2"]
    omega = float(x_opt[12])
    rpm = omega * 30.0 / np.pi
    lines = []
    lines.append("COAXIAL ROTOR OPTIMIZATION – SUMMARY")
    lines.append("===================================")
    lines.append("")
    lines.append(f"Mass [kg]                 : {cfg.mass_kg:.2f}")
    lines.append(f"Weight [N]                : {W:.2f}")
    lines.append(f"Altitude [m]              : {cfg.altitude_m:.2f}")
    lines.append(f"Design climb speed [m/s]  : {cfg.V_design:.2f}")
    lines.append(f"Disk loading [N/m^2]      : {cfg.disk_loading:.2f}")
    lines.append(f"Coaxial units             : {cfg.n_propulsors}")
    lines.append(f"Blades per rotor          : {cfg.n_blades}")
    lines.append("")
    lines.append(f"Target thrust/unit [N]    : {T_target:.2f}")
    lines.append(f"Achieved thrust [N]       : {float(final['totals']['T_total']):.2f}")
    lines.append(f"Shaft power [W]           : {float(final['P_shaft']):.2f}")
    lines.append(f"Omega [rad/s]             : {omega:.4f}")
    lines.append(f"RPM                       : {rpm:.2f}")
    lines.append(f"Tip Mach                  : {float(final['M_tip']):.4f} (limit {cfg.M_tip_max:.4f})")
    lines.append(f"Max c/R                   : {float(final['max_c_over_R']):.4f} (limit {cfg.MAX_C_OVER_R:.4f})")
    lines.append("")
    lines.append(f"Upper radius [m]          : {float(r1.radius):.4f}")
    lines.append(f"Lower radius [m]          : {float(r2.radius):.4f}")
    lines.append(f"Upper pitch root/tip [deg]: {float(r1.pitch[0]):.2f} / {float(r1.pitch[-1]):.2f}")
    lines.append(f"Lower pitch root/tip [deg]: {float(r2.pitch[0]):.2f} / {float(r2.pitch[-1]):.2f}")
    lines.append("")
    lines.append(f"Stage2 success            : {bool(stage2_result.success)}")
    lines.append(f"Stage2 message            : {stage2_result.message}")
    return "\n".join(lines)


def validate_npz_keys(npz_path: str, keys: list[str]):
    d = np.load(npz_path, allow_pickle=False)
    missing = [k for k in keys if k not in d.files]
    if missing:
        raise RuntimeError(f"NPZ missing required keys: {missing}")


# =============================================================================
# Main
# =============================================================================
def main():
    SKIP_STAGE_1 = True

    cfg = Config()

    # ---- Read inputs at beginning
    base_prop = propeller.Propeller.load_from_yaml(cfg.yaml_path)
    base_rot1 = copy.deepcopy(base_prop)
    base_rot2 = copy.deepcopy(base_prop)

    fl = fluid.Fluid(cfg.altitude_m)

    W = float(cfg.mass_kg * cfg.g)
    T_target = W / float(cfg.n_propulsors)

    r_R = radial_grid(cfg.n_stations, cfg.r_root_norm)

    # radius from disk loading (fixed Ru/Rl=1)
    Ru, _ = compute_equal_radii_from_DL(W=W, disk_loading=cfg.disk_loading, n_propulsors=cfg.n_propulsors)
    omega_max = cfg.M_tip_max * float(fl.a) / float(Ru)
    omega_min = 0.10 * omega_max

    print("=== DESIGN CASE ===")
    print(f"W={W:.1f} N, V_design={cfg.V_design:.2f} m/s, altitude={cfg.altitude_m:.0f} m")
    print(f"Target thrust per coax unit: {T_target:.2f} N")
    print(f"Fixed rotor radius: {Ru:.4f} m")
    print(f"Omega bounds: [{omega_min:.3f}, {omega_max:.3f}] rad/s")

    # ---- Bounds (13 vars)
    lb = [0.5]*4 + [0.5]*4 + [-5.0, -5.0] + [-5.0, -5.0] + [omega_min]
    ub = [2.2]*4 + [2.2]*4 + [45.0, 45.0] + [45.0, 45.0] + [omega_max]
    bounds = Bounds(lb, ub)

    # ---- Initial guess
    x0 = np.array(
        [1.0, 1.0, 1.0, 0.8,
         1.0, 1.0, 1.0, 0.8,
         25.0, 10.0,
         22.0, 9.0,
         0.70 * omega_max],
        dtype=float
    )

    # checkpoint load
    if os.path.exists(cfg.checkpoint_file):
        try:
            x_cp = np.load(cfg.checkpoint_file)
            if x_cp.shape == x0.shape:
                x0 = x_cp
                print(f"[INFO] Loaded checkpoint: {cfg.checkpoint_file}")
        except Exception as e:
            print(f"[WARN] Could not load checkpoint: {e}")
            
    # ---- Stage 1 (DE)
    if SKIP_STAGE_1 and os.path.exists(cfg.checkpoint_file):
        print("\n=== SKIPPING STAGE 1 (Using Checkpoint) ===")
        x1 = x0
    else:
        print("\n=== STAGE 1: Differential Evolution ===")

        obj1 = partial(
            objective_stage1,
            base_rot1=base_rot1, base_rot2=base_rot2,
            fl=fl, V_inf=cfg.V_design,
            r_R=r_R, cfg=cfg,
            T_target=T_target, W=W
            )

        def de_cb(xk, convergence=None):
            np.save(cfg.checkpoint_file, np.asarray(xk, dtype=float))
            return False
        
        res1 = differential_evolution(
            func=obj1,
            bounds=bounds,
            strategy="best1bin",
            maxiter=cfg.de_maxiter,
            popsize=cfg.de_popsize,
            tol=0.01,
            workers=-1,
            updating="deferred",
            disp=True,
            polish=False,
            callback=de_cb,
            )

        x1 = np.asarray(res1.x, dtype=float)



    # ---- Stage 2 (COBYLA)
    print("\n=== STAGE 2: COBYLA Polish ===")
    cache = EvalCache()

    def obj2(x):
        res = cache.get(x, base_rot1=base_rot1, base_rot2=base_rot2, fl=fl, V_inf=cfg.V_design, r_R=r_R, cfg=cfg, iters=cfg.stage2_coupling_iters, W=W)
        P = float(res["P_shaft"])
        print(f"COBYLA step: P = {P:.1f} W", flush=True)
        if (not np.isfinite(P)) or P <= 1.0:
            return 1e12
        return P

    cons = [
        {"type": "ineq", "fun": lambda x: c_thrust_min(x, cache=cache, base_rot1=base_rot1, base_rot2=base_rot2, fl=fl, V_inf=cfg.V_design, r_R=r_R, cfg=cfg, T_target=T_target, W=W)},
        {"type": "ineq", "fun": lambda x: c_thrust_max(x, cache=cache, base_rot1=base_rot1, base_rot2=base_rot2, fl=fl, V_inf=cfg.V_design, r_R=r_R, cfg=cfg, T_target=T_target, W=W)},
        {"type": "ineq", "fun": lambda x: c_mach(x, base_rot1=base_rot1, base_rot2=base_rot2, fl=fl, V_inf=cfg.V_design, r_R=r_R, cfg=cfg, W=W)},
        {"type": "ineq", "fun": lambda x: c_max_coverR(x, base_rot1=base_rot1, base_rot2=base_rot2, fl=fl, V_inf=cfg.V_design, r_R=r_R, cfg=cfg, W=W)},
        # analytic constraints (shape)
        {"type": "ineq", "fun": cU_c045_ge_c075},
        {"type": "ineq", "fun": cU_c075_ge_ctip},
        {"type": "ineq", "fun": cU_c045_ge_croot},
        {"type": "ineq", "fun": cL_c045_ge_c075},
        {"type": "ineq", "fun": cL_c075_ge_ctip},
        {"type": "ineq", "fun": cL_c045_ge_croot},
        {"type": "ineq", "fun": c_washout_upper},
        {"type": "ineq", "fun": c_washout_lower},
    ]
    # bounds as inequality constraints (COBYLA doesn't enforce Bounds)
    for i in range(len(x1)):
        cons.append({"type": "ineq", "fun": lambda x, i=i: float(x[i] - lb[i])})
        cons.append({"type": "ineq", "fun": lambda x, i=i: float(ub[i] - x[i])})

    res2 = minimize(
        fun=obj2,
        x0=x1,
        method="COBYLA",
        constraints=cons,
        options={"maxiter": int(cfg.stage2_maxiter), "rhobeg": 0.05, "disp": True},
    )

    x_opt = np.asarray(res2.x, dtype=float)
    np.save(cfg.checkpoint_file, x_opt)

    # ---- final strict evaluation
    final = evaluate_system(
        x_opt, base_rot1=base_rot1, base_rot2=base_rot2,
        fl=fl, V_inf=cfg.V_design, r_R=r_R, cfg=cfg,
        max_coupling_iter=cfg.strict_coupling_iters, W=W
    )
    print("\n=== FINAL DESIGN POINT ===")
    print(f"success={res2.success}, msg={res2.message}")
    print(f"T={final['totals']['T_total']:.2f} N, P={final['P_shaft']:.1f} W, RPM={x_opt[12]*30/np.pi:.1f}")
    print(f"M_tip={final['M_tip']:.3f} (limit {cfg.M_tip_max:.3f}), max(c/R)={final['max_c_over_R']:.3f}")

    # ---- sweep with RPM trim
    Vmin = int(np.floor(cfg.V_design - cfg.sweep_delta))
    Vmax = int(np.ceil(cfg.V_design + cfg.sweep_delta))
    Vs = np.arange(Vmin, Vmax + 1, cfg.sweep_step, dtype=float)

    x_geom = np.array(x_opt, dtype=float)
    omega_init = float(x_opt[12])

    P_shaft_sweep = []
    P_excess_sweep = []
    RPM_sweep = []
    Mtip_sweep = []
    T_sweep = []

    print("\n=== SWEEP (trim omega) ===")
    for V in Vs:
        omega_trim, res_trim = trim_omega_for_thrust(
            x_geom=x_geom, omega_init=omega_init, omega_min=omega_min, omega_max=omega_max,
            V_inf=float(V), base_rot1=base_rot1, base_rot2=base_rot2, fl=fl, r_R=r_R, cfg=cfg, W=W, T_target=T_target
        )
        omega_init = omega_trim

        T_here = float(res_trim["totals"]["T_total"])
        P_here = float(res_trim["P_shaft"])
        P_excess = float(P_here - T_here * float(V))

        P_shaft_sweep.append(P_here)
        P_excess_sweep.append(P_excess)
        RPM_sweep.append(float(omega_trim) * 30.0 / np.pi)
        Mtip_sweep.append(float(res_trim["M_tip"]))
        T_sweep.append(T_here)

        print(f"V={V:+.1f} | RPM={RPM_sweep[-1]:.1f} | T={T_here:.2f} | P={P_here:.1f} | Mtip={Mtip_sweep[-1]:.3f}")

    P_shaft_sweep = np.asarray(P_shaft_sweep, dtype=float)
    P_excess_sweep = np.asarray(P_excess_sweep, dtype=float)
    RPM_sweep = np.asarray(RPM_sweep, dtype=float)
    Mtip_sweep = np.asarray(Mtip_sweep, dtype=float)
    T_sweep = np.asarray(T_sweep, dtype=float)

    # ---- write outputs at end
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg.results_root, f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    # summary
    summary = build_summary(cfg, final, x_opt, T_target, W, res2)
    with open(os.path.join(run_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary)

    # YAML upper/lower
    import yaml
    upper_yaml_path = cfg.out_yaml_base + "_upper.yaml"
    lower_yaml_path = cfg.out_yaml_base + "_lower.yaml"
    with open(upper_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(rotor_to_yaml_dict(final["r1"]), f)
    with open(lower_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(rotor_to_yaml_dict(final["r2"]), f)

    # NPZ: all arrays needed for report plots
    npz_path = os.path.join(run_dir, "results.npz")
    np.savez(
        npz_path,
        # config scalars
        mass_kg=cfg.mass_kg, altitude_m=cfg.altitude_m, V_design=cfg.V_design,
        disk_loading=cfg.disk_loading, n_propulsors=cfg.n_propulsors, n_blades=cfg.n_blades,
        M_tip_max=cfg.M_tip_max, MAX_C_OVER_R=cfg.MAX_C_OVER_R, THRUST_TOL_N=cfg.THRUST_TOL_N,
        W=W, T_target=T_target,
        # design vector
        x_opt=x_opt,
        # geometry
        r_R=r_R,
        chordU=final["r1"].chord, chordL=final["r2"].chord,
        pitchU=final["r1"].pitch, pitchL=final["r2"].pitch,
        radiusU=final["r1"].radius, radiusL=final["r2"].radius,
        c_refU=final["r1"].c_ref, c_refL=final["r2"].c_ref,
        # distributions at design point
        phiU=final["out1"]["phi"], phiL=final["out2"]["phi"],
        dTdrU=final["out1"]["dT_dr"], dTdrL=final["out2"]["dT_dr"],
        alphaU=final["out1"]["alpha"], alphaL=final["out2"]["alpha"],
        VaxU=final["out1"]["Vax"], VaxL=final["out2"]["Vax"],
        V2_inflow=final["V2_inflow"],
        wU=final["w1"], wL=final["w2"],
        # totals at design point
        T_total=float(final["totals"]["T_total"]),
        Q_total=float(final["totals"]["Q_total"]),
        P_shaft=float(final["P_shaft"]),
        M_tip=float(final["M_tip"]),
        max_c_over_R=float(final["max_c_over_R"]),
        # sweep
        Vs=Vs,
        P_shaft_sweep=P_shaft_sweep,
        P_excess_sweep=P_excess_sweep,
        RPM_sweep=RPM_sweep,
        Mtip_sweep=Mtip_sweep,
        T_sweep=T_sweep,
    )

    # minimal validation: ensure NPZ contains keys needed for plots
    validate_npz_keys(npz_path, [
        "r_R", "chordU", "chordL", "pitchU", "pitchL",
        "phiU", "phiL", "dTdrU", "dTdrL",
        "Vs", "P_shaft_sweep", "P_excess_sweep",
        "RPM_sweep", "Mtip_sweep"
    ])

    # plots
    make_plots_from_npz(npz_path, run_dir, cfg)

    print("\n=== OUTPUTS ===")
    print(f"Run folder : {run_dir}")
    print(f"NPZ        : {npz_path}")
    print(f"Summary    : {os.path.join(run_dir, 'summary.txt')}")
    print(f"YAML upper : {upper_yaml_path}")
    print(f"YAML lower : {lower_yaml_path}")
    print("PNGs       : fig_*.png in run folder")


if __name__ == "__main__":
    main()
