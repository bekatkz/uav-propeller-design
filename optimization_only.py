# -*- coding: utf-8 -*-
"""
optimization_only.py

Optimization + data export (NO plotting code inside).

At the end it can optionally call plotting_coaxial.make_required_plots,
but that is a single-line call that you can comment out if you want a strict separation.

Outputs
-------
- results/run_YYYYMMDD_HHMMSS/results.npz  (all data needed to recreate plots)
- results/run_YYYYMMDD_HHMMSS/summary.txt
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
import numpy as np

from scipy.optimize import differential_evolution, minimize, brentq
from scipy.interpolate import PchipInterpolator

import fluid
import bemt_coaxial
import plotting_coaxial  # optional: just to generate plots at the end
import yaml


@dataclass(frozen=True)
class Config:
    # Aircraft & design state
    mass_kg: float = 650.0
    g: float = 9.81
    altitude_m: float = 500.0
    V_design: float = 3.0

    # Rotor system
    n_coax_units: int = 8
    disk_loading: float = 160.0  # N/m^2 for coaxial total area (both rotors)
    n_blades: int = 2
    airfoil_id: int = 2412

    # Discretization
    n_stations: int = 30
    r_root_norm: float = 0.10

    # Coaxial model parameters
    wake_factor: float = 2.0

    # Optimization
    de_maxiter: int = 50    
    de_popsize: int = 20
    stage2_maxiter: int = 1200

    # Constraints / limits
    thrust_tol_N: float = 5.0
    max_c_over_R: float = 0.25

    # Tip Mach bounds for omega
    M_tip_start: float = 0.40
    M_tip_min: float = 0.15
    M_tip_max: float = 0.70

    # Sweep
    sweep_delta: int = 10
    sweep_step: int = 1

    # Outputs
    results_root: str = "results"
    checkpoint_file: str = "checkpoint_updated.npy"
    
    #AoA
    alpha_max_deg: float = 12.0



def radial_grid(n_stations: int, r_root_norm: float) -> np.ndarray:
    return np.linspace(float(r_root_norm), 1.0, int(n_stations))


def rotor_radius_from_disk_loading(*, T_unit: float, disk_loading: float) -> float:
    return float(np.sqrt(float(T_unit) / (2.0 * np.pi * float(disk_loading))))


def chord_from_ctrl_c_over_R(r_R: np.ndarray, cR_ctrl: np.ndarray) -> np.ndarray:
    r0 = float(r_R[0])
    r_pts = np.array([r0, 0.45, 0.75, 1.0], dtype=float)
    c_pts = np.array(cR_ctrl, dtype=float)
    cR = PchipInterpolator(r_pts, c_pts)(r_R)
    return np.maximum(cR, 1e-5)


def twist_from_ctrl_deg(r_R: np.ndarray, beta_ctrl_deg: np.ndarray) -> np.ndarray:
    r0 = float(r_R[0])
    r_pts = np.array([r0, 0.30, 0.60, 0.85, 1.0], dtype=float)
    b_pts = np.array(beta_ctrl_deg, dtype=float)
    return PchipInterpolator(r_pts, b_pts)(r_R)


def enforce_monotone_washout(beta_ctrl_deg: np.ndarray) -> float:
    b = np.asarray(beta_ctrl_deg, dtype=float)
    diffs = np.diff(b)  # should be <= 0
    v = np.maximum(diffs, 0.0)
    return float(np.sum(v * v))


def enforce_chord_taper(cR_ctrl: np.ndarray) -> float:
    """
    Forces the chord to taper (shrink) towards the tip.
    Penalizes the optimizer if an outer control point is wider than the previous one.
    """
    c = np.asarray(cR_ctrl, dtype=float)
    # We check the difference between the 2nd, 3rd, and 4th control points
    diffs = np.diff(c[1:]) 
    v = np.maximum(diffs, 0.0)  # >0 means the chord incorrectly grew larger
    return float(np.sum(v * v))

def evaluate_design(x: np.ndarray, *, cfg: Config, fl: fluid.Fluid, r_R: np.ndarray, T_target_unit: float, V_inf: float) -> dict:
    x = np.asarray(x, dtype=float)
    cR_ctrl = x[0:4]
    betaU_ctrl = x[4:9]
    betaL_ctrl = x[9:14]
    omega = float(x[14])

    R = rotor_radius_from_disk_loading(T_unit=T_target_unit, disk_loading=cfg.disk_loading)

    c_over_R = chord_from_ctrl_c_over_R(r_R, cR_ctrl)
    chord_m = c_over_R * R

    twistU_deg = twist_from_ctrl_deg(r_R, betaU_ctrl)
    twistL_deg = twist_from_ctrl_deg(r_R, betaL_ctrl)

    rho = float(fl.rho)
    a_sound = float(fl.a)
    mu = float(fl.nu * fl.rho)

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

    return {
        "R": float(R),
        "omega": float(omega),
        "M_tip": float(omega * R / a_sound),
        "c_over_R": np.asarray(c_over_R, dtype=float),
        "chord_m": np.asarray(chord_m, dtype=float),
        "twistU_deg": np.asarray(twistU_deg, dtype=float),
        "twistL_deg": np.asarray(twistL_deg, dtype=float),
        "T": float(out["totals"]["T"][0]),
        "P": float(out["totals"]["P"][0]),
        "upper": out["upper"],
        "lower": out["lower"],
    }


def objective_stage1(x: np.ndarray, *, cfg: Config, fl: fluid.Fluid, r_R: np.ndarray, T_target_unit: float, V_inf: float) -> float:
    res = evaluate_design(x, cfg=cfg, fl=fl, r_R=r_R, T_target_unit=T_target_unit, V_inf=V_inf)
    
    P = float(res["P"])
    if (not np.isfinite(P)) or P <= 1.0:
        return 1e12
    
    # ---- AoA "realism" and SMOOTHNESS penalties ----
    aU = np.asarray(res["upper"]["alpha_deg"], dtype=float)
    aL = np.asarray(res["lower"]["alpha_deg"], dtype=float)
    alpha_lim = float(cfg.alpha_max_deg)
    
    # 1. Prevent Stall: Penalize exceeding max AoA limit
    viol = np.maximum(0.0, np.abs(aU) - alpha_lim)**2 + np.maximum(0.0, np.abs(aL) - alpha_lim)**2
    penalty_alpha_limit = 2e3 * float(np.mean(viol)) * max(P, 1.0)

    # 2. NEW - Flatten the AoA: Penalize variance (waviness) to force a stable, flat line
    penalty_alpha_var = 5e3 * (float(np.var(aU)) + float(np.var(aL))) * max(P, 1.0)

    # ---- Thrust constraint ----
    # ---- Thrust constraint ----
    T = float(res["T"])
    shortfall = max(0.0, (T_target_unit - T) / max(T_target_unit, 1e-9))
    over = max(0.0, (T - (T_target_unit + cfg.thrust_tol_N)) / max(T_target_unit, 1e-9))
    
    # Fix: Removed '* P' and massively increased the weight. 
    # The optimizer MUST hit the thrust target now, no excuses.
    penalty_T = (shortfall * shortfall + over * over) * 1e8

    # ---- Geometric Constraints (Forces a smooth, realistic blade) ----
    cR_max = float(np.max(res["c_over_R"]))
    penalty_cR = (max(0.0, cR_max - cfg.max_c_over_R) ** 2) * 5e4 * P

    cR_ctrl = np.asarray(x, float)[0:4]
    betaU_ctrl = np.asarray(x, float)[4:9]
    betaL_ctrl = np.asarray(x, float)[9:14]
    
    # 3. STRICT Washout: Multiply by 10x so the optimizer can't ignore it
    washout_pen = 2e4 * (enforce_monotone_washout(betaU_ctrl) + enforce_monotone_washout(betaL_ctrl)) * max(P, 1.0)
    
    # 4. NEW - Chord Taper: Force the blade to become thinner at the tip
    chord_taper_pen = 5e4 * enforce_chord_taper(cR_ctrl) * max(P, 1.0)

    # Sum all penalties with the power calculation
    return float(P + penalty_T + penalty_cR + washout_pen + chord_taper_pen + penalty_alpha_limit + penalty_alpha_var)

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
    x_geom = np.asarray(x_geom, dtype=float)

    def thrust_err(omega: float) -> tuple[float, dict]:
        xx = x_geom.copy()
        xx[14] = float(omega)
        res = evaluate_design(xx, cfg=cfg, fl=fl, r_R=r_R, T_target_unit=T_target_unit, V_inf=V_inf)
        return float(res["T"] - T_target_unit), res

    lo = max(float(omega_lo), float(omega_init) / 1.6)
    hi = min(float(omega_hi), float(omega_init) * 1.6)

    f_lo, _ = thrust_err(lo)
    f_hi, _ = thrust_err(hi)

    expand = 0
    while f_lo * f_hi > 0.0 and expand < 10:
        lo = max(float(omega_lo), lo / 1.6)
        hi = min(float(omega_hi), hi * 1.6)
        f_lo, _ = thrust_err(lo)
        f_hi, _ = thrust_err(hi)
        expand += 1

    if f_lo * f_hi > 0.0:
        omega_pick = lo if abs(f_lo) < abs(f_hi) else hi
        _, res_pick = thrust_err(omega_pick)
        return float(omega_pick), res_pick

    def f(om: float) -> float:
        val, _ = thrust_err(om)
        return float(val)

    omega_trim = float(brentq(f, lo, hi, xtol=1e-6, rtol=1e-8, maxiter=80))
    _, res_trim = thrust_err(omega_trim)
    return omega_trim, res_trim


def save_results_to_yaml(filepath: str, cfg: Config, final: dict, r_R: np.ndarray):
    """
    Saves the optimized propeller geometry and design state to a YAML file.
    """
    data = {
        "design_state": {
            "mass_kg": float(cfg.mass_kg),
            "altitude_m": float(cfg.altitude_m),
            "climb_speed_m_s": float(cfg.V_design),
            "thrust_target_N": float(final["T"]),
            "shaft_power_W": float(final["P"])
        },
        "rotor_geometry": {
            "n_blades": int(cfg.n_blades),
            "radius_m": float(final["R"]),
            "airfoil_id": int(cfg.airfoil_id),
            "omega_rad_s": float(final["omega"]),
            "RPM": float(final["omega"] * 30.0 / np.pi)
        },
        "blade_distributions": {
            "stations_r_R": [float(x) for x in r_R],
            "chord_m": [float(x) for x in final["chord_m"]],
            "chord_over_R": [float(x) for x in final["c_over_R"]],
            "twist_upper_deg": [float(x) for x in final["twistU_deg"]],
            "twist_lower_deg": [float(x) for x in final["twistL_deg"]]
        }
    }
    
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def main():
    cfg = Config()

    # ---- Reads at beginning
    fl = fluid.Fluid(cfg.altitude_m)
    W = float(cfg.mass_kg * cfg.g)
    T_target_unit = float(W / cfg.n_coax_units)

    r_R = radial_grid(cfg.n_stations, cfg.r_root_norm)
    R = rotor_radius_from_disk_loading(T_unit=T_target_unit, disk_loading=cfg.disk_loading)

    omega_lo = float(cfg.M_tip_min * fl.a / R)
    omega_hi = float(cfg.M_tip_max * fl.a / R)
    omega_start = float(cfg.M_tip_start * fl.a / R)

    # bounds (15 vars)
    # bounds (15 vars)
    lb = (
        # c/R control points [r0, 0.45, 0.75, 1.0]
        [0.05, 0.05, 0.03, 0.015] +
        
        # upper twist control points [r0, 0.30, 0.60, 0.85, 1.0] (deg)
        [10.0, 10.0,  8.0,  6.0,  4.0] +
        
        # lower twist ctrl points (deg)
        [ 8.0,  8.0,  7.0,  5.0,  3.0] +
        
        # omega
        [omega_lo]
        )
    ub = (
        # chord: cap peak chord & reduce weird bulges
        [0.16, 0.16, 0.12, 0.08] +
        
        # upper twist: prevent crazy root pitch
        [35.0, 30.0, 24.0, 18.0, 14.0] +
        
        # lower twist: typically a bit lower because it sits in wake
        [32.0, 28.0, 22.0, 17.0, 13.0] +
        
        # omega
        [omega_hi]
        )


    x0 = np.array(
        [0.10, 0.14, 0.10, 0.06,
         28.0, 22.0, 16.0, 12.0, 10.0,
         26.0, 20.0, 15.0, 11.0,  9.0,
         omega_start],
        dtype=float
    )

   # if os.path.exists(cfg.checkpoint_file):
    #    try:
     #       xcp = np.load(cfg.checkpoint_file)
      #      if xcp.shape == x0.shape:
       #         x0 = xcp
        #        print(f"[INFO] Loaded checkpoint: {cfg.checkpoint_file}")
        #except Exception as e:
          #  print(f"[WARN] Could not load checkpoint: {e}")

    # ---- Stage 1: DE
    print("=== STAGE 1: Differential Evolution ===")
    bounds = list(zip(lb, ub))
    obj1 = lambda x: objective_stage1(x, cfg=cfg, fl=fl, r_R=r_R, T_target_unit=T_target_unit, V_inf=cfg.V_design)

    def de_cb(xk, convergence=None):
       # np.save(cfg.checkpoint_file, np.asarray(xk, dtype=float))
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
        workers=1,
        callback=de_cb,
    )
    x1 = np.asarray(res1.x, dtype=float)

    # ---- Stage 2: COBYLA (polish)
    print("\n=== STAGE 2: COBYLA Polish ===")
    cons = []
    # thrust band + c/R cap (inequality g(x) >= 0)
    def c_T_min(x): return evaluate_design(x, cfg=cfg, fl=fl, r_R=r_R, T_target_unit=T_target_unit, V_inf=cfg.V_design)["T"] - T_target_unit
    def c_T_max(x): return (T_target_unit + cfg.thrust_tol_N) - evaluate_design(x, cfg=cfg, fl=fl, r_R=r_R, T_target_unit=T_target_unit, V_inf=cfg.V_design)["T"]
    def c_cR(x): return cfg.max_c_over_R - np.max(evaluate_design(x, cfg=cfg, fl=fl, r_R=r_R, T_target_unit=T_target_unit, V_inf=cfg.V_design)["c_over_R"])
    cons += [{"type":"ineq","fun":c_T_min},{"type":"ineq","fun":c_T_max},{"type":"ineq","fun":c_cR}]
    # bounds as constraints
    for i in range(len(x1)):
        cons.append({"type":"ineq","fun": lambda x, i=i: float(x[i] - lb[i])})
        cons.append({"type":"ineq","fun": lambda x, i=i: float(ub[i] - x[i])})

    def c_alpha_upper(x):
        res = evaluate_design(x, cfg=cfg, fl=fl, r_R=r_R, T_target_unit=T_target_unit, V_inf=cfg.V_design)
        aU = np.asarray(res["upper"]["alpha_deg"], dtype=float)
        return float(cfg.alpha_max_deg - np.max(np.abs(aU)))

    def c_alpha_lower(x):
        res = evaluate_design(x, cfg=cfg, fl=fl, r_R=r_R, T_target_unit=T_target_unit, V_inf=cfg.V_design)
        aL = np.asarray(res["lower"]["alpha_deg"], dtype=float)
        return float(cfg.alpha_max_deg - np.max(np.abs(aL)))

    cons += [{"type":"ineq","fun": c_alpha_upper}, {"type":"ineq","fun": c_alpha_lower}]


    def obj2(x):
        res = evaluate_design(x, cfg=cfg, fl=fl, r_R=r_R, T_target_unit=T_target_unit, V_inf=cfg.V_design)
        P = float(res["P"]); T = float(res["T"])
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
    #np.save(cfg.checkpoint_file, x_opt)

    # ---- Final eval at design point
    final = evaluate_design(x_opt, cfg=cfg, fl=fl, r_R=r_R, T_target_unit=T_target_unit, V_inf=cfg.V_design)

    # ---- Sweep with trim
    Vmin = int(np.floor(cfg.V_design - cfg.sweep_delta))
    Vmax = int(np.ceil(cfg.V_design + cfg.sweep_delta))
    Vs = np.arange(Vmin, Vmax + 1, cfg.sweep_step, dtype=float)

    x_geom = x_opt.copy()
    omega_init = float(x_opt[14])

    P_sweep, P_excess_sweep, RPM_sweep, Mtip_sweep, T_sweep = [], [], [], [], []
    for V in Vs:
        omega_trim, res_trim = trim_omega_for_thrust(
            x_geom=x_geom, cfg=cfg, fl=fl, r_R=r_R, T_target_unit=T_target_unit,
            V_inf=float(V), omega_lo=omega_lo, omega_hi=omega_hi, omega_init=omega_init
        )
        omega_init = omega_trim
        T_here = float(res_trim["T"])
        P_here = float(res_trim["P"])
        P_sweep.append(P_here)
        P_excess_sweep.append(P_here - T_here * float(V))
        RPM_sweep.append(float(omega_trim) * 30.0 / np.pi)
        Mtip_sweep.append(float(res_trim["M_tip"]))
        T_sweep.append(T_here)

    P_sweep = np.asarray(P_sweep, float)
    P_excess_sweep = np.asarray(P_excess_sweep, float)
    RPM_sweep = np.asarray(RPM_sweep, float)
    Mtip_sweep = np.asarray(Mtip_sweep, float)
    T_sweep = np.asarray(T_sweep, float)

    # ---- Writes at end
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg.results_root, f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    # summary
    summary_lines = [
        "COAXIAL OPTIMIZATION SUMMARY",
        "===========================",
        f"Altitude [m]              : {cfg.altitude_m:.2f}",
        f"Design climb speed [m/s]  : {cfg.V_design:.2f}",
        f"Disk loading [N/m^2]      : {cfg.disk_loading:.2f}",
        f"Target thrust/unit [N]    : {T_target_unit:.2f}",
        "",
        f"Achieved thrust [N]       : {final['T']:.2f}",
        f"Total shaft power [W]     : {final['P']:.2f}",
        f"Omega [rad/s]             : {final['omega']:.6f}",
        f"RPM                       : {final['omega']*30/np.pi:.2f}",
        f"Tip Mach                  : {final['M_tip']:.4f}",
        f"Rotor radius [m]          : {final['R']:.4f}",
        f"Max c/R                   : {np.max(final['c_over_R']):.4f}",
        "",
        f"Stage2 success            : {bool(res2.success)}",
        f"Stage2 message            : {res2.message}",
    ]
    with open(os.path.join(run_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    upper = final["upper"]; lower = final["lower"]

    npz_path = os.path.join(run_dir, "results.npz")
    np.savez(
        npz_path,
        # config/scalars
        altitude_m=cfg.altitude_m, V_design=cfg.V_design,
        disk_loading=cfg.disk_loading, n_coax_units=cfg.n_coax_units,
        n_blades=cfg.n_blades, airfoil_id=cfg.airfoil_id, wake_factor=cfg.wake_factor,
        W=W, T_target_unit=T_target_unit,
        R=final["R"], omega=final["omega"], M_tip=final["M_tip"],
        # design vector + geometry
        x_opt=x_opt,
        r_R=r_R,
        c_over_R=final["c_over_R"],
        chord_m=final["chord_m"],
        twistU_deg=final["twistU_deg"],
        twistL_deg=final["twistL_deg"],
        # distributions at design point
        phiU_deg=np.degrees(upper["phi"]),
        phiL_deg=np.degrees(lower["phi"]),
        dTdrU=upper["dTdr"],
        dTdrL=lower["dTdr"],
        dQdrU=upper["dQdr"],
        dQdrL=lower["dQdr"],
        alphaU_deg=upper["alpha_deg"],
        alphaL_deg=lower["alpha_deg"],
        CL_upper=upper["CL"], CD_upper=upper["CD"],
        CL_lower=lower["CL"], CD_lower=lower["CD"],
        VaxU=upper["Vax"], VaxL=lower["Vax"],
        # totals at design point
        T_design=final["T"], P_design=final["P"],
        # sweep
        Vs=Vs,
        P_sweep=P_sweep,
        P_excess_sweep=P_excess_sweep,
        RPM_sweep=RPM_sweep,
        Mtip_sweep=Mtip_sweep,
        T_sweep=T_sweep,
    )

    # Generate YAML export
    yaml_path = os.path.join(run_dir, "optimized_propeller.yaml")
    try:
        save_results_to_yaml(yaml_path, cfg, final, r_R)
    except NameError:
        print("[WARN] Could not save YAML. Make sure 'import yaml' is at the top of the file.")

    # Optional: generate plots automatically
    plotting_coaxial.make_required_plots(npz_path, run_dir)

    print("\n=== OUTPUTS ===")
    print(f"Run folder : {run_dir}")
    print(f"NPZ        : {npz_path}")
    print(f"Summary    : {os.path.join(run_dir, 'summary.txt')}")
    print(f"YAML       : {yaml_path}")  # <--- Added to console output
    print("PNGs       : fig_*.png in run folder")


if __name__ == "__main__":
    main()
