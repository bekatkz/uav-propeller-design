# optimization_only.py
"""
Coaxial rotor optimization for minimum shaft power in hover / axial climb.

Design case
-----------
  Aircraft mass   : 650 kg
  Number of coaxial rotor units : 8  (each unit = upper + lower rotor)
  Blades per rotor: 2
  Airfoil         : NACA 2412 (via NeuralFoil / AeroSandbox)
  ISA altitude    : 500 m
  Design climb speed : 3 m/s
  Disk loading    : 160 N/m²  (total disk area per unit = πR²; both rotors share same R)

Optimization variables (19 total)
-----------------------------------
  x[0:4]    – upper chord control points c/R at [r_root, 0.45, 0.75, 1.0]
  x[4:8]    – lower chord control points c/R at [r_root, 0.45, 0.75, 1.0]
  x[8:13]   – upper rotor twist control points β [deg] at [r_root, 0.30, 0.60, 0.85, 1.0]
  x[13:18]  – lower rotor twist control points β [deg] at same knot locations
  x[18]     – tip angular velocity ω [rad/s] (shared tip speed)

Two-stage optimization
-----------------------
  Stage 1 – Differential Evolution (global search)
  Stage 2 – COBYLA (local refinement) on feasible design

Outputs
-------
  - results_*.npz in a run folder
  - fig_*.png plots
  - geometry.yaml (explicit geometry arrays for verification)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import yaml

from scipy.interpolate import PchipInterpolator
from scipy.optimize import differential_evolution, minimize

import fluid
import bemt_coaxial
import plotting_coaxial


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Aircraft / coaxial unit definition
    mass_kg: float = 650.0
    g: float = 9.80665
    n_coax_units: int = 8             # each unit has 2 rotors: upper+lower
    n_blades: int = 2

    # Design case
    altitude_m: float = 500.0
    V_design: float = 3.0             # climb speed [m/s]

    # Aerodynamics
    airfoil_id: int = 2412

    # Discretization
    n_stations: int = 30
    r_root_norm: float = 0.10         # 0.1R root cutout

    # Disk loading (per unit)
    disk_loading: float = 160.0       # N/m^2

    # Blade loading target (used as guidance / penalty)
    design_blade_loading: float = 0.1

    # Coaxial wake coupling
    wake_factor: float = 2.0          # far-wake factor heuristic

    # Tip Mach bounds (for omega bounds)
    M_tip_min: float = 0.25
    M_tip_max: float = 0.55
    M_tip_start: float = 0.40

    # Geometry constraints / penalties
    min_solidity: float = 0.03        # typical lower bound; tune as needed
    max_c_over_R: float = 0.18

    # Optimization
    de_max_iter: int = 60
    de_popsize: int = 16
    random_seed: int = 2

    # Trimming sweep
    V_sweep_min: float = -10.0
    V_sweep_max: float =  10.0
    V_sweep_step: float = 1.0

    # Output
    out_dir: str = "runs"
    checkpoint_file: str = "checkpoint_x.npy"


# ─────────────────────────────────────────────────────────────────────────────
# Utility / geometry mapping
# ─────────────────────────────────────────────────────────────────────────────

def radial_grid(n: int, r_root_norm: float) -> np.ndarray:
    """Non-dimensional radial grid r/R including root cutout to tip."""
    return np.linspace(float(r_root_norm), 1.0, int(n))


def rotor_radius_from_disk_loading(*, T_unit: float, disk_loading: float) -> float:
    """
    Rotor radius from disk loading for a coaxial unit.
    You stated disk loading is "calculated for each rotor -> distribute loading on 2xNrotor".
    In your code, T_target_unit already represents the thrust per coaxial unit.
    A coaxial unit has two rotors sharing the same disk area πR^2 (same radius).
    So we use disk_loading = T_unit / (πR^2) -> R = sqrt(T_unit/(π*DL)).
    """
    return float(np.sqrt(T_unit / (np.pi * disk_loading)))


def mean_solidity(B: int, chord_m: np.ndarray, R: float, r_R: np.ndarray) -> float:
    """Mean solidity σ̄ ≈ B/(πR) * mean(c)."""
    c_mean = float(np.mean(chord_m))
    return float(B * c_mean / (np.pi * R))


def chord_from_ctrl_c_over_R(r_R: np.ndarray, ctrl: np.ndarray) -> np.ndarray:
    """
    4-parameter chord spline in c/R using PCHIP.
    Knots at [r_root, 0.45, 0.75, 1.0] in r/R.
    """
    r0 = float(r_R[0])
    xk = np.array([r0, 0.45, 0.75, 1.0], dtype=float)
    yk = np.asarray(ctrl, dtype=float)
    return PchipInterpolator(xk, yk)(r_R)


def twist_from_ctrl_deg(r_R: np.ndarray, ctrl: np.ndarray) -> np.ndarray:
    """
    5-parameter twist spline in degrees using PCHIP.
    Knots at [r_root, 0.30, 0.60, 0.85, 1.0] in r/R.
    """
    r0 = float(r_R[0])
    xk = np.array([r0, 0.30, 0.60, 0.85, 1.0], dtype=float)
    yk = np.asarray(ctrl, dtype=float)
    return PchipInterpolator(xk, yk)(r_R)


def enforce_monotone_washout(beta_ctrl: np.ndarray) -> float:
    """
    Penalty for non-decreasing twist towards the tip.
    We want washout: β_root >= ... >= β_tip.
    Returns sum of positive violations.
    """
    beta_ctrl = np.asarray(beta_ctrl, dtype=float)
    diffs = np.diff(beta_ctrl)  # positive diffs => increases towards tip (bad)
    return float(np.sum(np.maximum(diffs, 0.0)))


# ─────────────────────────────────────────────────────────────────────────────
# Design evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_design(
    x: np.ndarray,
    *,
    cfg:           Config,
    fl:            fluid.Fluid,
    r_R:           np.ndarray,
    T_target_unit: float,
    V_inf:         float,
) -> dict:
    """
    Map the optimizer's parameter vector x to a physical design and evaluate it.

    Parameter vector (length 19)
    ----------------------------
      x[0:4]    : upper chord control points [c/R]
      x[4:8]    : lower chord control points [c/R]
      x[8:13]   : upper twist control points [deg]
      x[13:18]  : lower twist control points [deg]
      x[18]     : omega [rad/s] (shared tip speed)

    Returns
    -------
    dict with keys: R, omega, M_tip,
                    c_over_R_U, c_over_R_L, chordU_m, chordL_m,
                    twistU_deg, twistL_deg,
                    T, P, upper, lower,
                    solidity_U, solidity_L
    """
    x = np.asarray(x, dtype=float)

    # ── Unpack parameter vector ───────────────────────────────────────────────
    cRU_ctrl   = x[0:4]     # upper chord control points [c/R]
    cRL_ctrl   = x[4:8]     # lower chord control points [c/R]
    betaU_ctrl = x[8:13]    # upper twist control points [deg]
    betaL_ctrl = x[13:18]   # lower twist control points [deg]
    omega      = float(x[18])

    # ── Geometry ──────────────────────────────────────────────────────────────
    R          = rotor_radius_from_disk_loading(T_unit=T_target_unit, disk_loading=cfg.disk_loading)

    c_over_R_U = chord_from_ctrl_c_over_R(r_R, cRU_ctrl)
    c_over_R_L = chord_from_ctrl_c_over_R(r_R, cRL_ctrl)

    chordU_m   = c_over_R_U * R
    chordL_m   = c_over_R_L * R

    twistU_deg = twist_from_ctrl_deg(r_R, betaU_ctrl)
    twistL_deg = twist_from_ctrl_deg(r_R, betaL_ctrl)

    # ── Fluid properties ──────────────────────────────────────────────────────
    rho     = float(fl.rho)
    a_sound = float(fl.a)
    mu      = float(fl.nu * fl.rho)

    # ── BEMT evaluation ───────────────────────────────────────────────────────
    out = bemt_coaxial.coaxial_bemt_fixed(
        rho             = rho,
        mu              = mu,
        a_sound         = a_sound,
        T_total_target  = float(T_target_unit),
        R               = float(R),
        B               = int(cfg.n_blades),
        omega           = float(omega),
        chord_upper     = chordU_m,
        chord_lower     = chordL_m,
        twist_upper_deg = twistU_deg,
        twist_lower_deg = twistL_deg,
        V_inf           = float(V_inf),
        airfoil         = int(cfg.airfoil_id),
        wake_factor     = float(cfg.wake_factor),
        n_stations      = int(cfg.n_stations),
        r_root_cutout   = float(cfg.r_root_norm),
    )

    solidity_U = mean_solidity(cfg.n_blades, chordU_m, R, r_R)
    solidity_L = mean_solidity(cfg.n_blades, chordL_m, R, r_R)

    return {
        "R":           float(R),
        "omega":       float(omega),
        "M_tip":       float(omega * R / a_sound),

        "c_over_R_U":  c_over_R_U,
        "c_over_R_L":  c_over_R_L,
        "chordU_m":    chordU_m,
        "chordL_m":    chordL_m,

        "twistU_deg":  twistU_deg,
        "twistL_deg":  twistL_deg,

        "T":           float(out["totals"]["T"][0]),
        "P":           float(out["totals"]["P"][0]),
        "upper":       out["upper"],
        "lower":       out["lower"],

        "solidity_U":  float(solidity_U),
        "solidity_L":  float(solidity_L),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Optimization objectives
# ─────────────────────────────────────────────────────────────────────────────

def objective_stage1(
    x: np.ndarray,
    *,
    cfg: Config,
    fl: fluid.Fluid,
    r_R: np.ndarray,
    T_target_unit: float,
    V_inf: float,
) -> float:
    """
    Stage 1 objective:
    Minimize shaft power + soft penalties for unphysical geometry.
    """
    res = evaluate_design(x, cfg=cfg, fl=fl, r_R=r_R, T_target_unit=T_target_unit, V_inf=V_inf)

    P = float(res["P"])
    T = float(res["T"])

    # base objective: power
    J = P

    # penalty: thrust mismatch (should be close; DE doesn't trim)
    J += 5e3 * (T - T_target_unit) ** 2 / max(T_target_unit**2, 1.0)

    # penalty: too large chord anywhere
    cR_max = float(max(np.max(res["c_over_R_U"]), np.max(res["c_over_R_L"])))
    if cR_max > cfg.max_c_over_R:
        J += 1e6 * (cR_max - cfg.max_c_over_R) ** 2

    # penalty: low solidity (either rotor)
    sol_min = min(float(res["solidity_U"]), float(res["solidity_L"]))
    sol_deficit = max(0.0, cfg.min_solidity - sol_min)
    J += 2e6 * sol_deficit**2

    # washout penalty (monotone decreasing twist towards the tip)
    x_arr = np.asarray(x, dtype=float)
    washout_pen = (
        enforce_monotone_washout(x_arr[8:13]) +
        enforce_monotone_washout(x_arr[13:18])
    )
    J += 2e3 * washout_pen * max(P, 1.0)

    return float(J)


def objective_stage2(
    x: np.ndarray,
    *,
    cfg: Config,
    fl: fluid.Fluid,
    r_R: np.ndarray,
    T_target_unit: float,
    V_inf: float,
) -> float:
    """
    Stage 2 objective: minimize power with geometry fixed, but trimmed omega handled outside.
    """
    res = evaluate_design(x, cfg=cfg, fl=fl, r_R=r_R, T_target_unit=T_target_unit, V_inf=V_inf)
    return float(res["P"])


# ─────────────────────────────────────────────────────────────────────────────
# Trimming omega for sweep (geometry fixed)
# ─────────────────────────────────────────────────────────────────────────────

def trim_omega_for_thrust(
    x_geom: np.ndarray,
    *,
    cfg:           Config,
    fl:            fluid.Fluid,
    r_R:           np.ndarray,
    T_target_unit: float,
    V_inf:         float,
    omega_lo:      float,
    omega_hi:      float,
    omega_init:    float,
) -> tuple[float, dict]:
    """
    Find the rotational speed ω that achieves exactly T_target_unit at V_inf,
    holding the blade geometry fixed.

    Method: simple bracket expansion + bisection on thrust residual.

    Parameters
    ----------
    x_geom     : optimizer parameter vector with initial omega at x[18]
    omega_init : starting estimate for ω (typically the previous trimmed value)

    Returns
    -------
    (omega_trim, result_dict)
    """
    x_geom = np.asarray(x_geom, dtype=float)

    def thrust_err(omega: float):
        xx      = x_geom.copy()
        xx[18]  = float(omega)
        res     = evaluate_design(xx, cfg=cfg, fl=fl, r_R=r_R,
                                  T_target_unit=T_target_unit, V_inf=V_inf)
        return float(res["T"] - T_target_unit), res

    lo = max(float(omega_lo), float(omega_init) / 1.6)
    hi = min(float(omega_hi), float(omega_init) * 1.6)

    f_lo, _ = thrust_err(lo)
    f_hi, _ = thrust_err(hi)

    for _ in range(10):
        if f_lo * f_hi <= 0.0:
            break
        lo = max(float(omega_lo), lo / 1.6)
        hi = min(float(omega_hi), hi * 1.6)
        f_lo, _ = thrust_err(lo)
        f_hi, _ = thrust_err(hi)

    # If still no sign change, just clamp to nearest bound
    if f_lo * f_hi > 0.0:
        if abs(f_lo) < abs(f_hi):
            _, res = thrust_err(lo)
            return lo, res
        _, res = thrust_err(hi)
        return hi, res

    # Bisection
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f_mid, res_mid = thrust_err(mid)
        if abs(f_mid) < 1e-6 * max(T_target_unit, 1.0):
            return mid, res_mid
        if f_lo * f_mid <= 0.0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid

    # best effort
    mid = 0.5 * (lo + hi)
    _, res = thrust_err(mid)
    return mid, res


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cfg = Config()

    # ════════════════════════════════════════════════════════════════════════
    # READ – all file I/O at the start
    # ════════════════════════════════════════════════════════════════════════
    fl             = fluid.Fluid(cfg.altitude_m)
    W              = float(cfg.mass_kg * cfg.g)
    T_target_unit  = float(W / cfg.n_coax_units)

    r_R = radial_grid(cfg.n_stations, cfg.r_root_norm)
    R   = rotor_radius_from_disk_loading(T_unit=T_target_unit, disk_loading=cfg.disk_loading)

    # Tip-speed bounds from Mach limits
    omega_lo    = float(cfg.M_tip_min * fl.a / R)
    omega_hi    = float(cfg.M_tip_max * fl.a / R)
    omega_start = float(cfg.M_tip_start * fl.a / R)

    # Minimum chord based on blade loading heuristic (very rough)
    # You may tune this; it mainly prevents vanishing chord
    c_over_R_min = max(0.03, cfg.design_blade_loading * 0.5)

    # Initial guess and bounds (19 variables)
    lb = (
        # upper c/R knots at [r_root, 0.45, 0.75, 1.0]
        [c_over_R_min,       c_over_R_min,       c_over_R_min * 0.8, c_over_R_min * 0.5] +
        # lower c/R knots
        [c_over_R_min,       c_over_R_min,       c_over_R_min * 0.8, c_over_R_min * 0.5] +
        # upper twist knots [deg] at [r_root, 0.30, 0.60, 0.85, 1.0]
        [10.0, 10.0,  8.0,  6.0,  4.0] +
        # lower twist knots [deg]
        [ 8.0,  8.0,  7.0,  5.0,  3.0] +
        # omega [rad/s]
        [omega_lo]
    )

    ub = (
        # upper c/R knots
        [0.16, 0.16, 0.14, 0.10] +
        # lower c/R knots
        [0.16, 0.16, 0.14, 0.10] +
        # upper twist knots
        [35.0, 30.0, 24.0, 18.0, 14.0] +
        # lower twist knots
        [32.0, 28.0, 22.0, 17.0, 13.0] +
        # omega
        [omega_hi]
    )

    x0 = np.array(
        # upper chord
        [max(c_over_R_min, 0.10), 0.14, 0.11, 0.07,
         # lower chord (start similar)
         max(c_over_R_min, 0.10), 0.14, 0.11, 0.07,
         # upper twist
         28.0, 22.0, 16.0, 12.0, 10.0,
         # lower twist
         26.0, 20.0, 15.0, 11.0,  9.0,
         # omega
         omega_start],
        dtype=float,
    )

    # ── Load checkpoint if available ──────────────────────────────────────────
    if os.path.exists(cfg.checkpoint_file):
        try:
            xcp = np.load(cfg.checkpoint_file)
            if xcp.shape == x0.shape:
                x0 = xcp
                print(f"[INFO] Loaded checkpoint from {cfg.checkpoint_file}")
        except Exception as exc:
            print(f"[WARN] Could not load checkpoint: {exc}")

    # ════════════════════════════════════════════════════════════════════════
    # Stage 1 – Differential Evolution (global search)
    # ════════════════════════════════════════════════════════════════════════
    print("\n=== STAGE 1: Differential Evolution ===")
    bounds = list(zip(lb, ub))
    obj1   = lambda x: objective_stage1(
        x, cfg=cfg, fl=fl, r_R=r_R, T_target_unit=T_target_unit, V_inf=cfg.V_design
    )

    de_res = differential_evolution(
        obj1,
        bounds=bounds,
        maxiter=int(cfg.de_max_iter),
        popsize=int(cfg.de_popsize),
        seed=int(cfg.random_seed),
        polish=False,
        disp=True,
    )

    x_best = np.asarray(de_res.x, dtype=float)
    np.save(cfg.checkpoint_file, x_best)

    # ════════════════════════════════════════════════════════════════════════
    # Stage 2 – COBYLA (local refinement)
    # ════════════════════════════════════════════════════════════════════════
    print("\n=== STAGE 2: COBYLA ===")
    obj2 = lambda x: objective_stage2(
        x, cfg=cfg, fl=fl, r_R=r_R, T_target_unit=T_target_unit, V_inf=cfg.V_design
    )

    # Simple box constraints for COBYLA via inequality functions
    cons = []
    for i in range(len(lb)):
        cons.append({"type": "ineq", "fun": lambda x, i=i: x[i] - lb[i]})
        cons.append({"type": "ineq", "fun": lambda x, i=i: ub[i] - x[i]})

    cob_res = minimize(
        obj2,
        x0=x_best,
        method="COBYLA",
        constraints=cons,
        options={"maxiter": 400, "rhobeg": 0.5, "disp": True},
    )

    x_final = np.asarray(cob_res.x, dtype=float)

    # Evaluate final at design point
    final = evaluate_design(
        x_final, cfg=cfg, fl=fl, r_R=r_R, T_target_unit=T_target_unit, V_inf=cfg.V_design
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Sweep of climb speeds: trim ω to match thrust at each V
    # ─────────────────────────────────────────────────────────────────────────
    V_sweep = np.arange(cfg.V_sweep_min, cfg.V_sweep_max + 0.5 * cfg.V_sweep_step, cfg.V_sweep_step, dtype=float)

    RPM_sweep = np.zeros_like(V_sweep)
    P_sweep   = np.zeros_like(V_sweep)

    P_excl_climb = np.zeros_like(V_sweep)

    omega_prev = float(final["omega"])
    x_geom = x_final.copy()

    for k, V in enumerate(V_sweep):
        omega_trim, resV = trim_omega_for_thrust(
            x_geom,
            cfg=cfg, fl=fl, r_R=r_R, T_target_unit=T_target_unit, V_inf=float(V),
            omega_lo=omega_lo, omega_hi=omega_hi, omega_init=omega_prev,
        )
        omega_prev = float(omega_trim)
        x_geom[18] = omega_prev

        RPM_sweep[k] = omega_prev * 30.0 / np.pi
        P_sweep[k]   = float(resV["P"])

        # excluding climb power: P - T*V (sign convention: climb positive)
        P_excl_climb[k] = float(resV["P"] - resV["T"] * float(V))

    # ─────────────────────────────────────────────────────────────────────────
    # WRITE – all file outputs at the end
    # ─────────────────────────────────────────────────────────────────────────
    os.makedirs(cfg.out_dir, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg.out_dir, f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    npz_path = os.path.join(run_dir, "results.npz")

    # Store arrays needed by plotting_coaxial
    np.savez(
        npz_path,
        # scalars / config
        altitude_m     = float(cfg.altitude_m),
        V_design       = float(cfg.V_design),
        airfoil_id     = int(cfg.airfoil_id),
        disk_loading   = float(cfg.disk_loading),
        wake_factor    = float(cfg.wake_factor),

        # radial grid
        r_R            = r_R,

        # final geometry
        R              = float(final["R"]),
        omega          = float(final["omega"]),
        M_tip          = float(final["M_tip"]),
        solidity_U     = float(final["solidity_U"]),
        solidity_L     = float(final["solidity_L"]),

        c_over_R_U     = final["c_over_R_U"],
        c_over_R_L     = final["c_over_R_L"],
        chordU_m       = final["chordU_m"],
        chordL_m       = final["chordL_m"],
        twistU_deg     = final["twistU_deg"],
        twistL_deg     = final["twistL_deg"],

        # distributions from BEMT
        phiU           = final["upper"]["phi"],
        phiL           = final["lower"]["phi"],
        alphaU         = final["upper"]["alpha"],
        alphaL         = final["lower"]["alpha"],
        dTdrU          = final["upper"]["dTdr"],
        dTdrL          = final["lower"]["dTdr"],

        # totals
        T_total        = float(final["T"]),
        P_total        = float(final["P"]),

        # sweep
        V_sweep        = V_sweep,
        RPM_sweep      = RPM_sweep,
        P_sweep        = P_sweep,
        P_excl_climb   = P_excl_climb,
    )

    # ── YAML geometry export (for inspection) ────────────────────────────────
    yaml_path = os.path.join(run_dir, "geometry.yaml")
    geom = {
        "meta": {
            "altitude_m": float(cfg.altitude_m),
            "V_design_mps": float(cfg.V_design),
            "disk_loading_N_m2": float(cfg.disk_loading),
            "airfoil": f"NACA{int(cfg.airfoil_id)}",
            "n_blades": int(cfg.n_blades),
            "n_stations": int(cfg.n_stations),
            "r_root_norm": float(cfg.r_root_norm),
            "wake_factor": float(cfg.wake_factor),
        },
        "design_point": {
            "T_target_unit_N": float(T_target_unit),
            "T_achieved_N": float(final["T"]),
            "P_shaft_W": float(final["P"]),
            "omega_rad_s": float(final["omega"]),
            "rpm": float(final["omega"] * 30.0 / np.pi),
            "tip_mach": float(final["M_tip"]),
            "R_m": float(final["R"]),
            "solidity_upper": float(final["solidity_U"]),
            "solidity_lower": float(final["solidity_L"]),
        },
        "grid": {
            "r_over_R": [float(v) for v in r_R],
        },
        "geometry": {
            "c_over_R_upper": [float(v) for v in final["c_over_R_U"]],
            "c_over_R_lower": [float(v) for v in final["c_over_R_L"]],
            "chord_m_upper":  [float(v) for v in final["chordU_m"]],
            "chord_m_lower":  [float(v) for v in final["chordL_m"]],
            "twist_deg_upper": [float(v) for v in final["twistU_deg"]],
            "twist_deg_lower": [float(v) for v in final["twistL_deg"]],
        },
    }
    with open(yaml_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(geom, fh, sort_keys=False)

    # ── Plots ────────────────────────────────────────────────────────────────
    plotting_coaxial.make_required_plots(npz_path, run_dir)

    print("\n=== OUTPUTS ===")
    print(f"  Run folder : {run_dir}")
    print(f"  NPZ        : {os.path.join(run_dir, 'results.npz')}")
    print("  PNGs       : fig_*.png in run folder")
    print("  YAML       : geometry.yaml in run folder")
    
    
    from access_clcd import _cached_base
    print(_cached_base.cache_info())


if __name__ == "__main__":
    main()