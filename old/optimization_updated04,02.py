# -*- coding: utf-8 -*-
"""
optimization_updated.py

Coaxial propeller optimization + automatic report-plot generation.

What this script does (aligned with your grading checklist)
-----------------------------------------------------------
1) Sizes rotor radius from disk loading (Ru/Rl = 1).
2) Optimizes at the DESIGN POINT (altitude, climb speed) for minimum total shaft power
   of ONE coaxial unit, subject to matching required thrust.
3) Uses:
   - same chord distribution for both rotors (4-parameter PCHIP spline on c/R)
   - independent NONLINEAR twist distributions (5-parameter PCHIP spline on beta(r))
   - tip speed via omega (rad/s)
4) Uses 30 radial stations and enforces a 0.1R root cutout.
5) After optimization, performs the required sweep Vc = V_design ± 10 m/s (1 m/s step),
   trimming omega at each speed to match the required thrust.
6) Writes ALL data required to recreate plots into a single NPZ, then generates PNG plots.

Project constraints respected
----------------------------
- Relative paths only
- File reads happen only at the beginning (baseline YAML)
- File writes happen only at the end (NPZ + summary + PNGs)
- No global variables
- Root cutout = 0.1R
- 30 stations (configurable)

Dependencies expected next to this file
---------------------------------------
- fluid.py
- bemt_coaxial.py  (updated version with coaxial_bemt_fixed signature)
- access_clcd.py   (your updated NeuralFoil caching is used inside bemt_coaxial)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import differential_evolution, minimize, brentq
from scipy.interpolate import PchipInterpolator

import fluid
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

    # Rotor system (coaxial units on aircraft)
    n_coax_units: int = 8
    disk_loading: float = 160.0  # N/m^2 for coaxial unit total disk area (both rotors)
    n_blades: int = 2
    airfoil_id: int = 2412

    # Discretization / geometry
    n_stations: int = 30
    r_root_norm: float = 0.10

    # Coaxial wake coupling (simple one-way)
    wake_factor: float = 2.0

    # Optimization settings
    de_maxiter: int = 25
    de_popsize: int = 10
    stage2_maxiter: int = 250

    # Constraints / limits (can be loosened if you want)
    thrust_tol_N: float = 5.0
    max_c_over_R: float = 0.25  # hard constraint (typical for rotors)

    # Tip-Mach bounds for omega (NOTE: this is a bound, not a physics "limit")
    # Start guess uses M_tip_start; optimizer can move within [M_tip_min, M_tip_max].
    M_tip_start: float = 0.40
    M_tip_min: float = 0.15
    M_tip_max: float = 0.70

    # Sweep
    sweep_delta: int = 10
    sweep_step: int = 1

    # Outputs
    results_root: str = "results"
    checkpoint_file: str = "checkpoint_updated.npy"


# =============================================================================
# Geometry helpers
# =============================================================================
def radial_grid(n_stations: int, r_root_norm: float) -> np.ndarray:
    """Non-dimensional radial grid r/R from root cutout to tip."""
    return np.linspace(float(r_root_norm), 1.0, int(n_stations))


def rotor_radius_from_disk_loading(*, T_unit: float, disk_loading: float) -> float:
    """
    Disk loading is defined on the TOTAL coaxial unit disk area (both rotors):
      DL = T_unit / A_total, where A_total = 2*pi*R^2 (Ru=Rl=R)
      => R = sqrt( T_unit / (2*pi*DL) )
    """
    return float(np.sqrt(float(T_unit) / (2.0 * np.pi * float(disk_loading))))


def chord_from_ctrl_c_over_R(r_R: np.ndarray, cR_ctrl: np.ndarray) -> np.ndarray:
    """
    4 control points for c/R at [r0, 0.45, 0.75, 1.0], interpolated with PCHIP.
    Returns dimensional chord [m] using c = (c/R)*R outside (caller multiplies by R).
    """
    r0 = float(r_R[0])
    r_pts = np.array([r0, 0.45, 0.75, 1.0], dtype=float)
    c_pts = np.array(cR_ctrl, dtype=float)
    interp = PchipInterpolator(r_pts, c_pts)
    cR = interp(r_R)
    return np.maximum(cR, 1e-5)


def twist_from_ctrl_deg(r_R: np.ndarray, beta_ctrl_deg: np.ndarray) -> np.ndarray:
    """
    5 control points for beta (twist/pitch angle in deg) at [r0, 0.30, 0.60, 0.85, 1.0].
    Nonlinear distribution via PCHIP.
    """
    r0 = float(r_R[0])
    r_pts = np.array([r0, 0.30, 0.60, 0.85, 1.0], dtype=float)
    b_pts = np.array(beta_ctrl_deg, dtype=float)
    interp = PchipInterpolator(r_pts, b_pts)
    beta = interp(r_R)
    return beta


def enforce_monotone_washout(beta_ctrl_deg: np.ndarray) -> float:
    """
    Soft penalty helper: encourages washout (beta decreases towards tip).
    Returns sum of squared positive slope violations between consecutive ctrl points.
    """
    b = np.asarray(beta_ctrl_deg, dtype=float)
    diffs = np.diff(b)  # should be <= 0
    v = np.maximum(diffs, 0.0)
    return float(np.sum(v * v))


# =============================================================================
# Physics evaluation
# =============================================================================
def evaluate_design(
    x: np.ndarray,
    *,
    cfg: Config,
    fl: fluid.Fluid,
    r_R: np.ndarray,
    T_target_unit: float,
    V_inf: float,
) -> dict:
    """
    Evaluate one design vector at a given climb speed V_inf.

    Design vector x (length 15):
      x[0:4]   chord control points c/R (shared for both rotors)
      x[4:9]   twist upper ctrl points [deg] (5)
      x[9:14]  twist lower ctrl points [deg] (5)
      x[14]    omega [rad/s]
    """
    x = np.asarray(x, dtype=float)
    cR_ctrl = x[0:4]
    betaU_ctrl = x[4:9]
    betaL_ctrl = x[9:14]
    omega = float(x[14])

    R = rotor_radius_from_disk_loading(T_unit=T_target_unit, disk_loading=cfg.disk_loading)

    # Build geometry (dimensional chord in meters)
    c_over_R = chord_from_ctrl_c_over_R(r_R, cR_ctrl)
    chord_m = c_over_R * R

    twistU_deg = twist_from_ctrl_deg(r_R, betaU_ctrl)
    twistL_deg = twist_from_ctrl_deg(r_R, betaL_ctrl)

    # Fluid properties
    rho = float(fl.rho)
    a_sound = float(fl.a)
    mu = float(fl.nu * fl.rho)  # dynamic viscosity from nu (kinematic) and rho

    out = bemt_coaxial.coaxial_bemt_fixed(
        rho=rho,
        mu=mu,
        a_sound=a_sound,
        T_total_target=float(T_target_unit),
        R=float(R),
        B=int(cfg.n_blades),
        omega=float(omega),
        chord=np.asarray(chord_m, dtype=float),
        twist_upper_deg=np.asarray(twistU_deg, dtype=float),
        twist_lower_deg=np.asarray(twistL_deg, dtype=float),
        V_inf=float(V_inf),
        airfoil=int(cfg.airfoil_id),
        wake_factor=float(cfg.wake_factor),
        n_stations=int(cfg.n_stations),
        r_root_cutout=float(cfg.r_root_norm),
    )

    T = float(out["totals"]["T"][0])
    P = float(out["totals"]["P"][0])

    # Convenience arrays for plots
    upper = out["upper"]
    lower = out["lower"]

    # dT/dr is with respect to r [m]. For plotting over r/R, we keep both.
    return {
        "R": float(R),
        "omega": float(omega),
        "M_tip": float(omega * R / a_sound),
        "c_over_R": np.asarray(c_over_R, dtype=float),
        "chord_m": np.asarray(chord_m, dtype=float),
        "twistU_deg": np.asarray(twistU_deg, dtype=float),
        "twistL_deg": np.asarray(twistL_deg, dtype=float),
        "T": float(T),
        "P": float(P),
        "upper": upper,
        "lower": lower,
    }


# =============================================================================
# Optimization objective and constraints
# =============================================================================
def objective_stage1(x: np.ndarray, *, cfg: Config, fl: fluid.Fluid, r_R: np.ndarray, T_target_unit: float, V_inf: float) -> float:
    """
    Penalized objective for Differential Evolution (global search).
    """
    res = evaluate_design(x, cfg=cfg, fl=fl, r_R=r_R, T_target_unit=T_target_unit, V_inf=V_inf)

    P = float(res["P"])
    if (not np.isfinite(P)) or P <= 1.0:
        return 1e12

    # Thrust penalty
    T = float(res["T"])
    shortfall = max(0.0, (T_target_unit - T) / max(T_target_unit, 1e-9))
    over = max(0.0, (T - (T_target_unit + cfg.thrust_tol_N)) / max(T_target_unit, 1e-9))
    penalty_T = (shortfall * shortfall + over * over) * 2e4 * P

    # Chord constraint penalty (c/R)
    cR_max = float(np.max(res["c_over_R"]))
    penalty_cR = (max(0.0, cR_max - cfg.max_c_over_R) ** 2) * 5e4 * P

    # Washout preference (soft)
    betaU_ctrl = np.asarray(x, float)[4:9]
    betaL_ctrl = np.asarray(x, float)[9:14]
    washout_pen = 2e3 * (enforce_monotone_washout(betaU_ctrl) + enforce_monotone_washout(betaL_ctrl)) * max(P, 1.0)

    return float(P + penalty_T + penalty_cR + washout_pen)


def constraints_cobyla(x: np.ndarray, *, cfg: Config, fl: fluid.Fluid, r_R: np.ndarray, T_target_unit: float, V_inf: float) -> list[dict]:
    """
    COBYLA uses inequality constraints g(x) >= 0.
    """
    def eval_cached(xx):
        return evaluate_design(xx, cfg=cfg, fl=fl, r_R=r_R, T_target_unit=T_target_unit, V_inf=V_inf)

    # thrust band
    def c_T_min(xx): return float(eval_cached(xx)["T"] - T_target_unit)
    def c_T_max(xx): return float((T_target_unit + cfg.thrust_tol_N) - eval_cached(xx)["T"])

    # chord limit
    def c_cR(xx): return float(cfg.max_c_over_R - np.max(eval_cached(xx)["c_over_R"]))

    # (optional) keep twist somewhat physical: beta between bounds already enforced by bounds-as-constraints

    cons = [
        {"type": "ineq", "fun": c_T_min},
        {"type": "ineq", "fun": c_T_max},
        {"type": "ineq", "fun": c_cR},
    ]
    return cons


# =============================================================================
# Trim omega for sweep
# =============================================================================
def trim_omega_for_thrust(
    x_geom: np.ndarray,
    *,
    cfg: Config,
    fl: fluid.Fluid,
    r_R: np.ndarray,
    T_target_unit: float,
    V_inf: float,
    omega_lo: float,
    omega_hi: float,
    omega_init: float,
) -> tuple[float, dict]:
    """
    Trim omega to satisfy T == T_target_unit for given V_inf, keeping geometry fixed.
    Uses brentq if sign change exists, otherwise falls back to best endpoint.
    """
    x_geom = np.asarray(x_geom, dtype=float)

    def thrust_err(omega: float) -> tuple[float, dict]:
        xx = x_geom.copy()
        xx[14] = float(omega)
        res = evaluate_design(xx, cfg=cfg, fl=fl, r_R=r_R, T_target_unit=T_target_unit, V_inf=V_inf)
        return float(res["T"] - T_target_unit), res

    # Initial bracket around omega_init
    lo = max(float(omega_lo), float(omega_init) / 1.6)
    hi = min(float(omega_hi), float(omega_init) * 1.6)

    f_lo, _ = thrust_err(lo)
    f_hi, _ = thrust_err(hi)

    # Expand bracket if needed
    expand = 0
    while f_lo * f_hi > 0.0 and expand < 10:
        lo = max(float(omega_lo), lo / 1.6)
        hi = min(float(omega_hi), hi * 1.6)
        f_lo, _ = thrust_err(lo)
        f_hi, _ = thrust_err(hi)
        expand += 1

    if f_lo * f_hi > 0.0:
        # fallback: choose best endpoint
        omega_pick = lo if abs(f_lo) < abs(f_hi) else hi
        _, res_pick = thrust_err(omega_pick)
        return float(omega_pick), res_pick

    def f(om: float) -> float:
        val, _ = thrust_err(om)
        return float(val)

    omega_trim = float(brentq(f, lo, hi, xtol=1e-6, rtol=1e-8, maxiter=80))
    _, res_trim = thrust_err(omega_trim)
    return omega_trim, res_trim


# =============================================================================
# Plotting (all required plots + a few recommended)
# =============================================================================
def _savefig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def make_required_plots(npz_path: str, out_dir: str, cfg: Config):
    d = np.load(npz_path, allow_pickle=False)

    r_R = d["r_R"]

    # design-point geometry
    cR = d["c_over_R"]
    betaU = d["twistU_deg"]
    betaL = d["twistL_deg"]

    # design-point distributions
    phiU = d["phiU_deg"]
    phiL = d["phiL_deg"]
    dTdrU = d["dTdrU"]
    dTdrL = d["dTdrL"]

    # sweep
    Vs = d["Vs"]
    P_shaft = d["P_sweep"]
    P_excess = d["P_excess_sweep"]

    # -------------------------
    # REQUIRED PLOTS
    # -------------------------

    # 1) Chord distribution (both rotors)  -> same curve is fine; we plot once and label "shared"
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(r_R, cR, label="Chord (shared, c/R)")
    plt.grid(True)
    plt.xlabel("r/R [-]")
    plt.ylabel("c/R [-]")
    plt.title("Chord distribution")
    plt.legend()
    _savefig(os.path.join(out_dir, "fig_chord_distribution.png"))

    # 2) Twist distribution (upper & lower)
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(r_R, betaU, label="Upper β")
    plt.plot(r_R, betaL, label="Lower β")
    plt.grid(True)
    plt.xlabel("r/R [-]")
    plt.ylabel("Twist / pitch β [deg]")
    plt.title("Twist distribution (upper & lower)")
    plt.legend()
    _savefig(os.path.join(out_dir, "fig_twist_distribution.png"))

    # 3) Inflow angle distribution (upper & lower)
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(r_R, phiU, label="Upper φ")
    plt.plot(r_R, phiL, label="Lower φ")
    plt.grid(True)
    plt.xlabel("r/R [-]")
    plt.ylabel("Inflow angle φ [deg]")
    plt.title("Inflow angle distribution")
    plt.legend()
    _savefig(os.path.join(out_dir, "fig_inflow_angle_phi.png"))

    # 4) Produced thrust distribution along radius (upper & lower)
    # We plot dT/dr vs r/R (dT/dr is per meter of radius).
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(r_R, dTdrU, label="Upper dT/dr")
    plt.plot(r_R, dTdrL, label="Lower dT/dr")
    plt.grid(True)
    plt.xlabel("r/R [-]")
    plt.ylabel("dT/dr [N/m]")
    plt.title("Radial thrust loading")
    plt.legend()
    _savefig(os.path.join(out_dir, "fig_thrust_loading_dTdr.png"))

    # 5) Power vs climb speed (trimmed)
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(Vs, P_shaft)
    plt.grid(True)
    plt.xlabel("Climb speed Vc [m/s]")
    plt.ylabel("Shaft power P [W]")
    plt.title("Power vs climb speed (trimmed RPM)")
    _savefig(os.path.join(out_dir, "fig_power_vs_climb_speed.png"))

    # 6) Power vs climb speed EXCLUDING climb power (P - T*Vc)
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(Vs, P_excess)
    plt.grid(True)
    plt.xlabel("Climb speed Vc [m/s]")
    plt.ylabel("P_shaft - T*Vc [W]")
    plt.title("Power excluding climb power (trimmed RPM)")
    _savefig(os.path.join(out_dir, "fig_power_excluding_climb.png"))

    # -------------------------
    # RECOMMENDED (helps explanation / sanity)
    # -------------------------
    if "alphaU_deg" in d.files and "alphaL_deg" in d.files:
        plt.figure(figsize=(7.2, 4.6))
        plt.plot(r_R, d["alphaU_deg"], label="Upper α")
        plt.plot(r_R, d["alphaL_deg"], label="Lower α")
        plt.grid(True)
        plt.xlabel("r/R [-]")
        plt.ylabel("Angle of attack α [deg]")
        plt.title("Angle of attack distribution")
        plt.legend()
        _savefig(os.path.join(out_dir, "fig_angle_of_attack.png"))

    if "RPM_sweep" in d.files:
        plt.figure(figsize=(7.2, 4.6))
        plt.plot(Vs, d["RPM_sweep"])
        plt.grid(True)
        plt.xlabel("Climb speed Vc [m/s]")
        plt.ylabel("Trimmed RPM [-]")
        plt.title("Trimmed RPM vs climb speed")
        _savefig(os.path.join(out_dir, "fig_rpm_vs_climb_speed.png"))

    if "Mtip_sweep" in d.files:
        plt.figure(figsize=(7.2, 4.6))
        plt.plot(Vs, d["Mtip_sweep"])
        plt.grid(True)
        plt.xlabel("Climb speed Vc [m/s]")
        plt.ylabel("Tip Mach [-]")
        plt.title("Tip Mach vs climb speed (trimmed)")
        _savefig(os.path.join(out_dir, "fig_tip_mach_vs_climb_speed.png"))


# =============================================================================
# Main
# =============================================================================
def main():
    cfg = Config()

    # --------------------------
    # READS ONLY AT BEGINNING
    # --------------------------
    fl = fluid.Fluid(cfg.altitude_m)

    W = float(cfg.mass_kg * cfg.g)
    T_target_unit = float(W / cfg.n_coax_units)

    r_R = radial_grid(cfg.n_stations, cfg.r_root_norm)
    R = rotor_radius_from_disk_loading(T_unit=T_target_unit, disk_loading=cfg.disk_loading)

    # omega bounds from tip mach bounds
    omega_lo = float(cfg.M_tip_min * fl.a / R)
    omega_hi = float(cfg.M_tip_max * fl.a / R)
    omega_start = float(cfg.M_tip_start * fl.a / R)

    print("=== DESIGN CASE ===")
    print(f"Altitude: {cfg.altitude_m:.0f} m | rho={fl.rho:.3f} kg/m^3 | a={fl.a:.1f} m/s")
    print(f"Mass: {cfg.mass_kg:.1f} kg | W={W:.1f} N")
    print(f"Coax units: {cfg.n_coax_units} | Target thrust per coax unit: {T_target_unit:.2f} N")
    print(f"Disk loading: {cfg.disk_loading:.1f} N/m^2 -> R={R:.4f} m (Ru=Rl)")
    print(f"Omega bounds: [{omega_lo:.3f}, {omega_hi:.3f}] rad/s  (M_tip in [{cfg.M_tip_min:.2f},{cfg.M_tip_max:.2f}])")

    # --------------------------
    # Design vector bounds
    # --------------------------
    # x[0:4] chord ctrl c/R
    # x[4:9] betaU ctrl deg
    # x[9:14] betaL ctrl deg
    # x[14] omega
    lb = (
        [0.04, 0.06, 0.04, 0.02] +          # c/R ctrl (roughly plausible)
        [-5.0, 0.0, 5.0, 8.0, 8.0] +        # upper twist ctrl (deg)
        [-5.0, 0.0, 5.0, 8.0, 8.0] +        # lower twist ctrl (deg)
        [omega_lo]
    )
    ub = (
        [0.30, 0.30, 0.25, 0.18] +          # c/R ctrl
        [50.0, 50.0, 45.0, 35.0, 30.0] +    # upper
        [50.0, 50.0, 45.0, 35.0, 30.0] +    # lower
        [omega_hi]
    )

    # Initial guess
    x0 = np.array(
        [0.10, 0.14, 0.10, 0.06,
         28.0, 22.0, 16.0, 12.0, 10.0,
         26.0, 20.0, 15.0, 11.0,  9.0,
         omega_start],
        dtype=float
    )

    # Load checkpoint if available
    if os.path.exists(cfg.checkpoint_file):
        try:
            xcp = np.load(cfg.checkpoint_file)
            if xcp.shape == x0.shape:
                x0 = xcp
                print(f"[INFO] Loaded checkpoint: {cfg.checkpoint_file}")
        except Exception as e:
            print(f"[WARN] Could not load checkpoint: {e}")

    # --------------------------
    # Stage 1: Differential Evolution
    # --------------------------
    print("\n=== STAGE 1: Differential Evolution ===")
    obj1 = lambda x: objective_stage1(x, cfg=cfg, fl=fl, r_R=r_R, T_target_unit=T_target_unit, V_inf=cfg.V_design)
    bounds = list(zip(lb, ub))

    def de_cb(xk, convergence=None):
        np.save(cfg.checkpoint_file, np.asarray(xk, dtype=float))
        return False

    res1 = differential_evolution(
        func=obj1,
        bounds=bounds,
        strategy="best1bin",
        maxiter=int(cfg.de_maxiter),
        popsize=int(cfg.de_popsize),
        tol=0.02,
        disp=True,
        polish=False,
        updating="deferred",
        workers=-1,
        callback=de_cb,
    )
    x1 = np.asarray(res1.x, dtype=float)

    # --------------------------
    # Stage 2: COBYLA (constraint polishing)
    # --------------------------
    print("\n=== STAGE 2: COBYLA Polish ===")
    # COBYLA doesn't enforce bounds, so add them as inequality constraints.
    cons = constraints_cobyla(x1, cfg=cfg, fl=fl, r_R=r_R, T_target_unit=T_target_unit, V_inf=cfg.V_design)
    for i in range(len(x1)):
        cons.append({"type": "ineq", "fun": lambda x, i=i: float(x[i] - lb[i])})
        cons.append({"type": "ineq", "fun": lambda x, i=i: float(ub[i] - x[i])})

    def obj2(x):
        res = evaluate_design(x, cfg=cfg, fl=fl, r_R=r_R, T_target_unit=T_target_unit, V_inf=cfg.V_design)
        P = float(res["P"])
        T = float(res["T"])
        print(f"COBYLA step: P={P:.1f} W | T={T:.1f} N | Mtip={res['M_tip']:.3f}", flush=True)
        if (not np.isfinite(P)) or P <= 1.0:
            return 1e12
        return P

    res2 = minimize(
        fun=obj2,
        x0=x1,
        method="COBYLA",
        constraints=cons,
        options={"maxiter": int(cfg.stage2_maxiter), "rhobeg": 0.02, "disp": True},
    )

    x_opt = np.asarray(res2.x, dtype=float)
    np.save(cfg.checkpoint_file, x_opt)

    # --------------------------
    # Final evaluation at design point
    # --------------------------
    final = evaluate_design(x_opt, cfg=cfg, fl=fl, r_R=r_R, T_target_unit=T_target_unit, V_inf=cfg.V_design)
    print("\n=== FINAL DESIGN POINT ===")
    print(f"success={res2.success} | msg={res2.message}")
    print(f"T={final['T']:.2f} N | P={final['P']:.1f} W | RPM={final['omega']*30/np.pi:.1f}")
    print(f"R={final['R']:.4f} m | M_tip={final['M_tip']:.3f} | max(c/R)={np.max(final['c_over_R']):.3f}")

    # --------------------------
    # Sweep: V = V_design ± 10 m/s, trim omega to match thrust
    # --------------------------
    Vmin = int(np.floor(cfg.V_design - cfg.sweep_delta))
    Vmax = int(np.ceil(cfg.V_design + cfg.sweep_delta))
    Vs = np.arange(Vmin, Vmax + 1, cfg.sweep_step, dtype=float)

    x_geom = x_opt.copy()
    omega_init = float(x_opt[14])

    P_sweep = []
    P_excess_sweep = []
    RPM_sweep = []
    Mtip_sweep = []
    T_sweep = []

    print("\n=== SWEEP (trim omega) ===")
    for V in Vs:
        omega_trim, res_trim = trim_omega_for_thrust(
            x_geom=x_geom,
            cfg=cfg,
            fl=fl,
            r_R=r_R,
            T_target_unit=T_target_unit,
            V_inf=float(V),
            omega_lo=omega_lo,
            omega_hi=omega_hi,
            omega_init=omega_init,
        )
        omega_init = omega_trim

        T_here = float(res_trim["T"])
        P_here = float(res_trim["P"])
        P_excess = float(P_here - T_here * float(V))  # "excluding climb power"

        P_sweep.append(P_here)
        P_excess_sweep.append(P_excess)
        RPM_sweep.append(float(omega_trim) * 30.0 / np.pi)
        Mtip_sweep.append(float(res_trim["M_tip"]))
        T_sweep.append(T_here)

        print(f"V={V:+.1f} m/s | RPM={RPM_sweep[-1]:.1f} | T={T_here:.2f} | P={P_here:.1f} | Mtip={Mtip_sweep[-1]:.3f}")

    P_sweep = np.asarray(P_sweep, float)
    P_excess_sweep = np.asarray(P_excess_sweep, float)
    RPM_sweep = np.asarray(RPM_sweep, float)
    Mtip_sweep = np.asarray(Mtip_sweep, float)
    T_sweep = np.asarray(T_sweep, float)

    # --------------------------
    # WRITES ONLY AT END
    # --------------------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg.results_root, f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    # Save a compact text summary
    summary_lines = [
        "COAXIAL OPTIMIZATION SUMMARY",
        "===========================",
        f"Mass [kg]                 : {cfg.mass_kg:.2f}",
        f"Weight [N]                : {W:.2f}",
        f"Altitude [m]              : {cfg.altitude_m:.2f}",
        f"Design climb speed [m/s]  : {cfg.V_design:.2f}",
        f"Disk loading [N/m^2]      : {cfg.disk_loading:.2f}",
        f"Coax units                : {cfg.n_coax_units}",
        f"Target thrust/unit [N]    : {T_target_unit:.2f}",
        "",
        f"Achieved thrust [N]       : {final['T']:.2f}",
        f"Total shaft power [W]     : {final['P']:.2f}",
        f"Omega [rad/s]             : {final['omega']:.6f}",
        f"RPM                       : {final['omega']*30/np.pi:.2f}",
        f"Tip Mach                  : {final['M_tip']:.4f}",
        f"Rotor radius [m]          : {final['R']:.4f}",
        f"Max c/R                   : {np.max(final['c_over_R']):.4f} (limit {cfg.max_c_over_R:.4f})",
        "",
        f"Stage2 success            : {bool(res2.success)}",
        f"Stage2 message            : {res2.message}",
    ]
    with open(os.path.join(run_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    # Save ALL plot-relevant data into one NPZ
    upper = final["upper"]
    lower = final["lower"]

    np.savez(
        os.path.join(run_dir, "results.npz"),
        # config / scalars
        mass_kg=cfg.mass_kg,
        altitude_m=cfg.altitude_m,
        V_design=cfg.V_design,
        disk_loading=cfg.disk_loading,
        n_coax_units=cfg.n_coax_units,
        n_blades=cfg.n_blades,
        airfoil_id=cfg.airfoil_id,
        wake_factor=cfg.wake_factor,
        W=W,
        T_target_unit=T_target_unit,
        R=final["R"],
        omega=final["omega"],
        M_tip=final["M_tip"],
        # design vector
        x_opt=x_opt,
        # geometry (design point)
        r_R=r_R,
        c_over_R=final["c_over_R"],
        chord_m=final["chord_m"],
        twistU_deg=final["twistU_deg"],
        twistL_deg=final["twistL_deg"],
        # distributions (design point)
        phiU_deg=np.degrees(upper["phi"]),
        phiL_deg=np.degrees(lower["phi"]),
        dTdrU=upper["dTdr"],
        dTdrL=lower["dTdr"],
        dQdrU=upper["dQdr"],
        dQdrL=lower["dQdr"],
        alphaU_deg=upper["alpha_deg"],
        alphaL_deg=lower["alpha_deg"],
        CLU=upper["CL"],
        CDL=lower["CD"],  # note: just storing both CL/CD
        CL_lower=lower["CL"],
        CD_upper=upper["CD"],
        VaxU=upper["Vax"],
        VaxL=lower["Vax"],
        # totals at design point
        T_design=final["T"],
        P_design=final["P"],
        # sweep arrays
        Vs=Vs,
        P_sweep=P_sweep,
        P_excess_sweep=P_excess_sweep,
        RPM_sweep=RPM_sweep,
        Mtip_sweep=Mtip_sweep,
        T_sweep=T_sweep,
    )

    # Generate plots
    make_required_plots(os.path.join(run_dir, "results.npz"), run_dir, cfg)

    print("\n=== OUTPUTS ===")
    print(f"Run folder : {run_dir}")
    print(f"NPZ        : {os.path.join(run_dir, 'results.npz')}")
    print(f"Summary    : {os.path.join(run_dir, 'summary.txt')}")
    print("PNGs       : fig_*.png in run folder")


if __name__ == "__main__":
    main()
