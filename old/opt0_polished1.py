# -*- coding: utf-8 -*-
"""
opt0_polished.py

COAXIAL PROPELLER OPTIMIZATION (Report-quality geometry)

This version is a cleaned, single-entry script that keeps the speed/robustness
of (DE -> COBYLA) while enforcing a manufacturable blade planform.

Design variables x (10)
----------------------
  x[0:4]  chord control multipliers at [r0, 0.45, 0.75, 1.0]   (nondimensional)
  x[4:6]  upper pitch: beta_root, beta_tip                    (deg)
  x[6:8]  lower pitch: beta_root, beta_tip                    (deg)
  x[8]    omega                                               (rad/s)
  x[9]    ratio = R_upper / R_lower                           (-)

Stages
------
Stage 1 (DE): global search with hinge penalties (fast physics).
Stage 2 (COBYLA): local polish with hard constraints and caching (tighter physics).

Why it produces "good looking" blades
-------------------------------------
• PCHIP interpolation: no spline overshoot / no hidden mid-span humps.
• Hard max(c/R) constraint: prevents paddle/brick planforms.
• Thrust tight band: prevents lazy over-thrusting solutions.

Notes
-----
• This keeps omega as a design variable (no internal trimming loop).
• bemt_coaxial.coaxial_bemt_fixed: we use max_coupling_iter only.
"""

from datetime import datetime
from dataclasses import dataclass
import os
import copy
import numpy as np
import matplotlib.pyplot as plt

from functools import partial
from scipy.optimize import minimize, Bounds, differential_evolution
from scipy.interpolate import PchipInterpolator

import fluid
import propeller
import bemt_coaxial


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    # --- Aircraft & flight condition ---
    mass_kg: float = 650.0
    g: float = 9.81
    altitude_m: float = 500.0
    V_design: float = 3.0

    # --- Rotor system ---
    n_propulsors: int = 8                 # number of coaxial units on the aircraft
    disk_loading: float = 160.0           # N/m^2 per coaxial unit area convention used in sizing
    n_blades: int = 2

    # --- Limits ---
    M_tip_max: float = 0.40

    # --- Geometry ---
    n_stations: int = 30
    r_root_norm: float = 0.10

    # Fixed physical chord scale in meters (realism enforced by MAX_C_OVER_R).
    c_ref_override_m: float = 0.15
    force_airfoil_id: int | None = 2412

    # --- Optimization settings ---
    de_maxiter: int = 15
    de_popsize: int = 10
    stage2_maxiter: int = 1000

    # --- Coupling iterations ---
    stage1_coupling_iters: int = 5
    stage2_coupling_iters: int = 15
    strict_coupling_iters: int = 25

    # --- Penalty weights (Stage 1) ---
    W_THRUST_SHORTFALL: float = 6000.0
    W_MACH: float = 2500.0
    W_SHAPE: float = 1500.0
    W_COVER_R: float = 12000.0

    # --- Hard constraints (Stage 2) ---
    MAX_C_OVER_R: float = 0.20
    THRUST_TOL_N: float = 5.0             # allowed overshoot above target in Stage 2

    # --- Bounds ---
    ratio_lb: float = 0.8
    ratio_ub: float = 1.2

    # --- Files ---
    yaml_path: str = os.path.join("data", "pybemt_tmotor28.yaml")
    out_yaml_base: str = os.path.join("data", "pybemt_optimized_ehang_polished")
    checkpoint_file: str = "checkpoint_opt0_polished.npy"


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
class EvalCache:
    """Cache to reuse the last expensive evaluation for objective + constraints."""
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


def chord_pchip(r_R: np.ndarray, c_ctrl: np.ndarray) -> np.ndarray:
    """Shape-preserving interpolation for chord control points."""
    r_pts = np.array([float(r_R[0]), 0.45, 0.75, 1.0], dtype=float)
    c_pts = np.array(c_ctrl, dtype=float)
    interp = PchipInterpolator(r_pts, c_pts)
    c = interp(r_R)
    return np.maximum(c, 1e-4)


def linear_pitch(r_R: np.ndarray, beta_root_deg: float, beta_tip_deg: float) -> np.ndarray:
    r0 = float(r_R[0])
    r1 = float(r_R[-1])
    t = (r_R - r0) / max(r1 - r0, 1e-12)
    return float(beta_root_deg) + (float(beta_tip_deg) - float(beta_root_deg)) * t


def compute_radius_pair_from_ratio(*, W: float, disk_loading: float, n_propulsors: int, ratio: float):
    """Total area per coaxial unit = (W/N_prop)/DL; split by ratio with A = pi(Ru^2 + Rl^2)."""
    A_total = (float(W) / float(n_propulsors)) / float(disk_loading)
    Rl = np.sqrt(A_total / (np.pi * (ratio**2 + 1.0)))
    Ru = ratio * Rl
    return float(Ru), float(Rl)


def override_scale_in_memory(prop, *, R_new: float, c_ref_new: float, nblades_new: int):
    prop.radius = float(R_new)
    prop.c_ref = float(c_ref_new)
    prop.nblades = int(nblades_new)
    return prop


# -----------------------------------------------------------------------------
# Shape helpers
# -----------------------------------------------------------------------------
def shape_violations(x: np.ndarray):
    """Nonnegative violations for taper + washout rules on control variables."""
    x = np.asarray(x, dtype=float)
    c_root, c_045, c_075, c_tip = x[0], x[1], x[2], x[3]
    b1_root, b1_tip = x[4], x[5]
    b2_root, b2_tip = x[6], x[7]

    # taper (prefer peak around 0.45R then decreasing)
    v1 = max(0.0, float(c_075 - c_045))
    v2 = max(0.0, float(c_tip - c_075))
    v3 = max(0.0, float(c_root - c_045))

    # washout
    v4 = max(0.0, float(b1_tip - b1_root))
    v5 = max(0.0, float(b2_tip - b2_root))
    return (v1, v2, v3, v4, v5)


def shape_penalty(x, P: float, cfg: Config) -> float:
    v = shape_violations(x)
    return float(cfg.W_SHAPE * sum(vi * vi for vi in v) * max(float(P), 1.0))


# -----------------------------------------------------------------------------
# Build rotors from design vector
# -----------------------------------------------------------------------------
def build_rotors_from_x(x, *, base_rot1, base_rot2, r_R, cfg: Config, W: float):
    x = np.asarray(x, dtype=float)

    c_ctrl = x[0:4]
    b1_root, b1_tip = float(x[4]), float(x[5])
    b2_root, b2_tip = float(x[6]), float(x[7])
    omega = float(x[8])
    ratio = float(x[9])

    Ru, Rl = compute_radius_pair_from_ratio(
        W=W, disk_loading=cfg.disk_loading, n_propulsors=cfg.n_propulsors, ratio=ratio
    )

    chord_dist = chord_pchip(r_R, c_ctrl)
    beta1 = linear_pitch(r_R, b1_root, b1_tip)
    beta2 = linear_pitch(r_R, b2_root, b2_tip)

    r1 = copy.deepcopy(base_rot1)
    r2 = copy.deepcopy(base_rot2)

    r1 = override_scale_in_memory(r1, R_new=Ru, c_ref_new=cfg.c_ref_override_m, nblades_new=cfg.n_blades)
    r2 = override_scale_in_memory(r2, R_new=Rl, c_ref_new=cfg.c_ref_override_m, nblades_new=cfg.n_blades)

    r1.r_R = np.array(r_R, dtype=float)
    r2.r_R = np.array(r_R, dtype=float)

    # Keep the same nondimensional chord distribution on both rotors (simple/robust)
    r1.chord = np.array(chord_dist, dtype=float)
    r2.chord = np.array(chord_dist, dtype=float)

    r1.pitch = np.array(beta1, dtype=float)
    r2.pitch = np.array(beta2, dtype=float)

    if cfg.force_airfoil_id is not None:
        r1.airfoil = np.full_like(r1.r_R, int(cfg.force_airfoil_id), dtype=int)
        r2.airfoil = np.full_like(r2.r_R, int(cfg.force_airfoil_id), dtype=int)

    r1.omega = omega
    r2.omega = omega
    return r1, r2


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
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

    omega = float(np.asarray(x, dtype=float)[8])
    P_shaft = float(totals["Q_total"] * omega)

    # For constraints/diagnostics
    c1 = np.max(np.asarray(r1.chord) * float(r1.c_ref))
    c2 = np.max(np.asarray(r2.chord) * float(r2.c_ref))
    max_c_over_R = max(c1 / float(r1.radius), c2 / float(r2.radius))

    Rmax = max(float(r1.radius), float(r2.radius))
    M_tip = float(omega) * Rmax / float(fl.a)

    return {
        "totals": totals,
        "out1": out1,
        "out2": out2,
        "coupling": coupling,
        "P_shaft": P_shaft,
        "r1": r1,
        "r2": r2,
        "max_c_over_R": float(max_c_over_R),
        "M_tip": float(M_tip),
    }


# -----------------------------------------------------------------------------
# Stage 1 objective (DE)
# -----------------------------------------------------------------------------
def objective_stage1(x, *, base_rot1, base_rot2, fl, V_inf, r_R, cfg: Config, T_target: float, W: float,
                     eval_counter=None, print_every: int = 100):
    if eval_counter is not None:
        eval_counter[0] += 1
        if (eval_counter[0] % int(print_every)) == 0:
            print(".", end="", flush=True)

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

    # hinge: thrust shortfall only
    shortfall = max(0.0, (T_target - T) / max(T_target, 1e-9))
    penalty_T = cfg.W_THRUST_SHORTFALL * (shortfall ** 2) * P

    # hinge: Mach only if violated
    mach_margin = cfg.M_tip_max - float(res["M_tip"])
    penalty_M = cfg.W_MACH * (max(0.0, -mach_margin) ** 2) * P

    # shape penalty on control variables
    penalty_shape = shape_penalty(x, P, cfg)

    # soft penalty for max(c/R) in stage 1 (guides DE away from paddles)
    v = max(0.0, float(res["max_c_over_R"]) - cfg.MAX_C_OVER_R)
    penalty_cR = cfg.W_COVER_R * (v ** 2) * P

    return float(P + penalty_T + penalty_M + penalty_shape + penalty_cR)


# -----------------------------------------------------------------------------
# Stage 2 (COBYLA): constraints
# COBYLA requires each constraint fun(x) >= 0
# -----------------------------------------------------------------------------
def c_thrust_min(x, *, cache: EvalCache, base_rot1, base_rot2, fl, V_inf, r_R, cfg: Config, T_target: float, W: float):
    res = cache.get(
        x, base_rot1=base_rot1, base_rot2=base_rot2, fl=fl, V_inf=V_inf, r_R=r_R, cfg=cfg, iters=cfg.stage2_coupling_iters, W=W
    )
    return float(res["totals"]["T_total"] - T_target)


def c_thrust_max(x, *, cache: EvalCache, base_rot1, base_rot2, fl, V_inf, r_R, cfg: Config, T_target: float, W: float):
    res = cache.get(
        x, base_rot1=base_rot1, base_rot2=base_rot2, fl=fl, V_inf=V_inf, r_R=r_R, cfg=cfg, iters=cfg.stage2_coupling_iters, W=W
    )
    return float((T_target + cfg.THRUST_TOL_N) - res["totals"]["T_total"])


def c_mach(x, *, base_rot1, base_rot2, fl, V_inf, r_R, cfg: Config, W: float):
    # cheap 1-iter eval (geometry + omega)
    res = evaluate_system(
        x, base_rot1=base_rot1, base_rot2=base_rot2, fl=fl, V_inf=V_inf, r_R=r_R, cfg=cfg, max_coupling_iter=1, W=W
    )
    return float(cfg.M_tip_max - res["M_tip"])


def c_max_coverR(x, *, base_rot1, base_rot2, fl, V_inf, r_R, cfg: Config, W: float):
    # cheap 1-iter eval (geometry only)
    res = evaluate_system(
        x, base_rot1=base_rot1, base_rot2=base_rot2, fl=fl, V_inf=V_inf, r_R=r_R, cfg=cfg, max_coupling_iter=1, W=W
    )
    return float(cfg.MAX_C_OVER_R - res["max_c_over_R"])


# Simple analytic shape constraints on control points
def c_washout_upper(x):
    x = np.asarray(x, dtype=float)
    return float(x[4] - x[5])  # beta_root - beta_tip >= 0


def c_washout_lower(x):
    x = np.asarray(x, dtype=float)
    return float(x[6] - x[7])


def c_c045_ge_c075(x):
    x = np.asarray(x, dtype=float)
    return float(x[1] - x[2])


def c_c075_ge_ctip(x):
    x = np.asarray(x, dtype=float)
    return float(x[2] - x[3])


def c_c045_ge_croot(x):
    x = np.asarray(x, dtype=float)
    return float(x[1] - x[0])


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_planform(rotor, title: str):
    r = np.asarray(rotor.r_R, dtype=float) * float(rotor.radius)
    c = np.asarray(rotor.chord, dtype=float) * float(rotor.c_ref)

    le = 0.25 * c
    te = -0.75 * c

    plt.figure(figsize=(10, 4))
    plt.fill_between(r, te, le, alpha=0.25)
    plt.plot(r, le, linewidth=2)
    plt.plot(r, te, linewidth=2)
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("Radius [m]")
    plt.ylabel("Chordwise [m]")
    plt.title(f"{title} | max chord={np.max(c):.3f} m | max(c/R)={np.max(c)/float(rotor.radius):.3f}")
    plt.tight_layout()
    plt.show()


def plot_distributions(r_R, r1, r2):
    fig, axs = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    axs[0].plot(r_R, r1.pitch, label="Upper")
    axs[0].plot(r_R, r2.pitch, label="Lower")
    axs[0].set_ylabel("Pitch [deg]")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(r_R, np.asarray(r1.chord) * float(r1.c_ref), label="Chord [m]")
    axs[1].set_xlabel("r/R [-]")
    axs[1].set_ylabel("Chord [m]")
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Results writer
# -----------------------------------------------------------------------------
def write_final_results_txt(
    *,
    cfg: Config,
    final: dict,
    x_opt: np.ndarray,
    T_target: float,
    W: float,
    stage2_result,
    filename: str | None = None,
):
    """
    Writes a clean, report-ready summary of the optimization result.
    Uses STRICT evaluation outputs (final dict).
    """
    if filename is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"opt_result_{ts}.txt"

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)

    totals = final["totals"]
    r1 = final["r1"]
    r2 = final["r2"]
    P_shaft = float(final["P_shaft"])
    M_tip = float(final["M_tip"])
    max_cR = float(final["max_c_over_R"])

    omega = float(x_opt[8])
    ratio = float(x_opt[9])

    n_rotors_total = int(cfg.n_propulsors * 2)

    with open(path, "w", encoding="utf-8") as f:
        f.write("COAXIAL ROTOR OPTIMIZATION – FINAL RESULT\n")
        f.write("=======================================\n\n")

        f.write("Run Information\n")
        f.write("----------------\n")
        f.write(f"Timestamp          : {datetime.now()}\n")
        f.write(f"Script             : {os.path.basename(__file__)}\n")
        f.write(f"Checkpoint file    : {cfg.checkpoint_file}\n")
        f.write(f"Stage 2 Success    : {stage2_result.success}\n")
        f.write(f"Stage 2 Message    : {stage2_result.message}\n\n")

        f.write("Mission & Inputs\n")
        f.write("----------------\n")
        f.write(f"Aircraft Mass      : {cfg.mass_kg:.1f} kg\n")
        f.write(f"Weight             : {W:.2f} N\n")
        f.write(f"Altitude           : {cfg.altitude_m:.1f} m\n")
        f.write(f"Design Climb Speed : {cfg.V_design:.2f} m/s\n")
        f.write(f"Disk Loading       : {cfg.disk_loading:.1f} N/m^2\n")
        f.write(f"Propulsors         : {cfg.n_propulsors} (coaxial)\n")
        f.write(f"Rotors Total       : {n_rotors_total}\n")
        f.write(f"Blades per Rotor   : {cfg.n_blades}\n")
        f.write(f"Tip Mach Limit     : {cfg.M_tip_max:.2f}\n")
        f.write(f"Root Cutout        : {cfg.r_root_norm:.2f} R\n\n")

        f.write("Optimization Result\n")
        f.write("-------------------\n")
        f.write(f"Target Thrust      : {T_target:.2f} N\n")
        f.write(f"Achieved Thrust    : {float(totals['T_total']):.2f} N\n")
        f.write(f"Power              : {P_shaft:.2f} W\n")
        f.write(f"Angular Speed      : {omega:.3f} rad/s\n")
        f.write(f"Tip Mach Number    : {M_tip:.4f}\n")
        f.write(f"Radius Ratio Ru/Rl : {ratio:.4f}\n\n")

        f.write("Geometry\n")
        f.write("--------\n")
        f.write(f"Upper Radius       : {float(r1.radius):.4f} m\n")
        f.write(f"Lower Radius       : {float(r2.radius):.4f} m\n")
        f.write(f"Max Chord          : {max(np.max(r1.chord*r1.c_ref), np.max(r2.chord*r2.c_ref)):.4f} m\n")
        f.write(f"Max c/R            : {max_cR:.4f}\n")
        f.write(f"Upper Pitch Root   : {float(r1.pitch[0]):.2f} deg\n")
        f.write(f"Upper Pitch Tip    : {float(r1.pitch[-1]):.2f} deg\n")
        f.write(f"Lower Pitch Root   : {float(r2.pitch[0]):.2f} deg\n")
        f.write(f"Lower Pitch Tip    : {float(r2.pitch[-1]):.2f} deg\n\n")

        f.write("Constraints\n")
        f.write("-----------\n")
        f.write(f"Thrust Band        : [{T_target:.2f}, {T_target + cfg.THRUST_TOL_N:.2f}] N\n")
        f.write(f"Mach Constraint    : {'OK' if M_tip <= cfg.M_tip_max else 'VIOLATED'}\n")
        f.write(f"Max c/R Constraint : {'OK' if max_cR <= cfg.MAX_C_OVER_R else 'VIOLATED'}\n\n")

        f.write("Optimization Metadata\n")
        f.write("---------------------\n")
        f.write("Stage 1            : Differential Evolution\n")
        f.write("Stage 2            : COBYLA\n")
        f.write(f"Stage 2 Evaluations: {stage2_result.nfev}\n")

    print(f"[RESULTS] Final results written to: {path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    cfg = Config()

    # Load baseline rotor
    base_prop = propeller.Propeller.load_from_yaml(cfg.yaml_path)
    base_rot1 = copy.deepcopy(base_prop)
    base_rot2 = copy.deepcopy(base_prop)

    fl = fluid.Fluid(cfg.altitude_m)

    W = float(cfg.mass_kg * cfg.g)
    T_target = W / float(cfg.n_propulsors)

    r_R = radial_grid(cfg.n_stations, cfg.r_root_norm)

    # Compute omega_max using worst-case radius across ratio bounds
    A_total = (W / cfg.n_propulsors) / cfg.disk_loading
    Rl_worst = np.sqrt(A_total / (np.pi * (cfg.ratio_ub**2 + 1.0)))
    Ru_worst = cfg.ratio_ub * Rl_worst
    Rmax_worst = max(Ru_worst, Rl_worst)
    omega_max = cfg.M_tip_max * float(fl.a) / float(Rmax_worst)

    # Bounds
    bounds = Bounds(
        [0.5]*4 + [-5.0]*4 + [0.2*omega_max] + [cfg.ratio_lb],
        [2.2]*4 + [45.0]*4 + [omega_max]     + [cfg.ratio_ub],
    )

    # Initial guess
    x0 = np.array([1.0, 1.0, 1.0, 0.8, 25.0, 10.0, 22.0, 9.0, 0.7*omega_max, 1.0], dtype=float)

    if os.path.exists(cfg.checkpoint_file):
        try:
            x_cp = np.load(cfg.checkpoint_file)
            if x_cp.shape == x0.shape:
                x0 = x_cp
                print(f"[INFO] Loaded checkpoint: {cfg.checkpoint_file}")
        except Exception as e:
            print(f"[WARN] Could not load checkpoint ({e}). Using default x0.")

    print("=== DESIGN CASE ===")
    print(f"Weight: {W:.2f} N | Altitude: {cfg.altitude_m:.1f} m | V_design: {cfg.V_design:.2f} m/s")
    print(f"Propulsors: {cfg.n_propulsors} | Blades per rotor: {cfg.n_blades}")
    print(f"T_target per coaxial unit: {T_target:.2f} N")
    print(f"Mach limit: {cfg.M_tip_max:.3f} | omega_max: {omega_max:.3f} rad/s")
    print(f"Hard constraint: max(c/R) <= {cfg.MAX_C_OVER_R:.3f}")
    print(f"Hard constraint: T in [{T_target:.2f}, {T_target + cfg.THRUST_TOL_N:.2f}] N")

    # ------------------------------------------------------------------
    # Stage 1: Differential Evolution (parallel safe via partial)
    # ------------------------------------------------------------------
    eval_counter = [0]
    obj1 = partial(
        objective_stage1,
        base_rot1=base_rot1,
        base_rot2=base_rot2,
        fl=fl,
        V_inf=cfg.V_design,
        r_R=r_R,
        cfg=cfg,
        T_target=T_target,
        W=W,
        eval_counter=eval_counter,
        print_every=80,
    )

    def de_checkpoint_callback(xk, convergence=None):
        # IMPORTANT: must return False; returning a truthy value stops DE early.
        np.save(cfg.checkpoint_file, np.asarray(xk, dtype=float))
        print(" [SAVED]", end="", flush=True)
        return False

    print("\n=== STAGE 1: Differential Evolution ===")
    res1 = differential_evolution(
        func=obj1,
        bounds=bounds,
        strategy="best1bin",
        maxiter=cfg.de_maxiter,
        popsize=cfg.de_popsize,
        tol=0.01,               # set to 0.0 if you want to force all generations
        workers=-1,
        updating="deferred",
        disp=True,
        polish=False,
        callback=de_checkpoint_callback,
    )
    x1 = np.asarray(res1.x, dtype=float)

    # ------------------------------------------------------------------
    # Stage 2: COBYLA polish with tight physics + caching
    # ------------------------------------------------------------------
    print("\n\n=== STAGE 2: COBYLA Polish ===")
    cache = EvalCache(atol=1e-9)

    def obj2(x):
        res = cache.get(
            x,
            base_rot1=base_rot1, base_rot2=base_rot2,
            fl=fl, V_inf=cfg.V_design, r_R=r_R, cfg=cfg,
            iters=cfg.stage2_coupling_iters, W=W
        )
        P = float(res["P_shaft"])
        if (not np.isfinite(P)) or P <= 1.0:
            return 1e12
        # tiny penalty if below target (should be prevented by constraints anyway)
        T = float(res["totals"]["T_total"])
        shortfall = max(0.0, (T_target - T) / max(T_target, 1e-9))
        return float(P * (1.0 + 0.05 * shortfall))

    lb, ub = bounds.lb, bounds.ub

    cons = [
        {"type": "ineq", "fun": lambda x: c_thrust_min(x, cache=cache, base_rot1=base_rot1, base_rot2=base_rot2, fl=fl, V_inf=cfg.V_design, r_R=r_R, cfg=cfg, T_target=T_target, W=W)},
        {"type": "ineq", "fun": lambda x: c_thrust_max(x, cache=cache, base_rot1=base_rot1, base_rot2=base_rot2, fl=fl, V_inf=cfg.V_design, r_R=r_R, cfg=cfg, T_target=T_target, W=W)},
        {"type": "ineq", "fun": lambda x: c_mach(x, base_rot1=base_rot1, base_rot2=base_rot2, fl=fl, V_inf=cfg.V_design, r_R=r_R, cfg=cfg, W=W)},
        {"type": "ineq", "fun": lambda x: c_max_coverR(x, base_rot1=base_rot1, base_rot2=base_rot2, fl=fl, V_inf=cfg.V_design, r_R=r_R, cfg=cfg, W=W)},
        # analytic shape constraints
        {"type": "ineq", "fun": c_washout_upper},
        {"type": "ineq", "fun": c_washout_lower},
        {"type": "ineq", "fun": c_c045_ge_c075},
        {"type": "ineq", "fun": c_c075_ge_ctip},
        {"type": "ineq", "fun": c_c045_ge_croot},
    ]

    # Explicit bounds for COBYLA
    for i in range(len(x1)):
        cons.append({"type": "ineq", "fun": lambda x, i=i: float(x[i] - lb[i])})
        cons.append({"type": "ineq", "fun": lambda x, i=i: float(ub[i] - x[i])})

    res2 = minimize(
        fun=obj2,
        x0=x1,
        method="COBYLA",
        constraints=cons,
        options={
            "maxiter": int(cfg.stage2_maxiter),
            "rhobeg": 0.05,
            "disp": True,
        },
    )

    x_opt = np.asarray(res2.x, dtype=float)
    np.save(cfg.checkpoint_file, x_opt)

    # ------------------------------------------------------------------
    # Final strict evaluation
    # ------------------------------------------------------------------
    print("\n=== FINAL STRICT EVALUATION ===")
    final = evaluate_system(
        x_opt,
        base_rot1=base_rot1,
        base_rot2=base_rot2,
        fl=fl,
        V_inf=cfg.V_design,
        r_R=r_R,
        cfg=cfg,
        max_coupling_iter=cfg.strict_coupling_iters,
        W=W,
    )

    totals = final["totals"]
    r1 = final["r1"]
    r2 = final["r2"]

    print("\n[FINAL RESULTS]")
    print(f"Stage2 success: {res2.success}")
    print(f"Stage2 message: {res2.message}")
    print(f"Thrust : {float(totals['T_total']):.2f} N (Target: {T_target:.2f} .. {T_target + cfg.THRUST_TOL_N:.2f} N)")
    print(f"Power  : {float(final['P_shaft']):.2f} W")
    print(f"omega  : {float(x_opt[8]):.4f} rad/s")
    print(f"ratio  : {float(x_opt[9]):.4f}")
    print(f"R_upper: {float(r1.radius):.4f} m")
    print(f"R_lower: {float(r2.radius):.4f} m")
    print(f"M_tip  : {float(final['M_tip']):.4f} (limit {cfg.M_tip_max:.4f})")
    print(f"max(c/R): {float(final['max_c_over_R']):.4f} (limit {cfg.MAX_C_OVER_R:.4f})")

    # ------------------------------------------------------------------
    # Write final results to text file (REPORT READY)
    # ------------------------------------------------------------------
    write_final_results_txt(
        cfg=cfg,
        final=final,
        x_opt=x_opt,
        T_target=T_target,
        W=W,
        stage2_result=res2,
    )

    # Save YAMLs
    out_upper = cfg.out_yaml_base + "_upper.yaml"
    out_lower = cfg.out_yaml_base + "_lower.yaml"
    r1.save_to_yaml(out_upper)
    r2.save_to_yaml(out_lower)
    print(f"\nSaved Upper Rotor to: {out_upper}")
    print(f"Saved Lower Rotor to: {out_lower}")

    # Plots
    plot_planform(r1, "Upper Rotor Planform (1:1)")
    plot_planform(r2, "Lower Rotor Planform (1:1)")
    plot_distributions(r_R, r1, r2)


if __name__ == "__main__":
    main()
