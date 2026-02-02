# opt0.py
"""
COAXIAL PROPELLER OPTIMIZATION (Option B: Shape-by-Construction) + Lower Rotor Chord Scale

What this version guarantees (by construction):
  - Twist is always washout (decreasing with radius) for both rotors.
  - Chord has a single hump then tapers (no wiggles).

What it adds (your requested improvement):
  - One extra variable: s2 = chord_scale_lower
    -> lower rotor chord = s2 * upper rotor chord (same shape, scaled)
    -> tight bounds (default 0.90..1.10) + small regularization to keep it near 1.0

Design variables x (11 total):
  Chord model (4):
    x[0] = c_root_nd      (nondimensional chord multiplier of c_ref)
    x[1] = c_peak_nd      (peak chord multiplier)
    x[2] = r_peak         (location of peak, in r/R)
    x[3] = c_tip_nd       (tip chord multiplier)

  Twist washout model (4):
    x[4] = beta1_root_deg (upper rotor root pitch)
    x[5] = d_beta1_deg    (upper rotor washout magnitude, >= 0)
    x[6] = beta2_root_deg (lower rotor root pitch)
    x[7] = d_beta2_deg    (lower rotor washout magnitude, >= 0)

  Operating & sizing (2):
    x[8]  = omega         (rad/s)
    x[9]  = ratio         (R_upper / R_lower)

  Coaxial asymmetry (1):
    x[10] = s2            (lower chord scale factor)

Optimization:
  Stage 1: Differential Evolution (global) with hinge penalties (T shortfall + Mach violation)
  Stage 2: COBYLA (local) with caching, constraints T>=target, Mach<=limit, and bounds-as-constraints

IMPORTANT:
  - This is a NEW parameterization. Old checkpoints are NOT compatible.
  - Delete checkpoint_opt0.npy before first run (recommended).
"""

from dataclasses import dataclass
import os
import copy
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, differential_evolution

import fluid
import propeller
import bemt_coaxial


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    # --- Aircraft & Flight Condition Inputs ---
    mass_kg: float = 650.0
    g: float = 9.81
    altitude_m: float = 500.0
    V_design: float = 3.0

    # --- Rotor Sizing & Limits ---
    disk_loading: float = 160.0
    M_tip_max: float = 0.40
    n_blades: int = 2
    n_propulsors: int = 8

    # --- Geometry Definition ---
    n_stations: int = 30
    r_root_norm: float = 0.10

    # --- File Paths ---
    yaml_path: str = os.path.join("data", "pybemt_tmotor28.yaml")
    out_yaml_path: str = os.path.join("data", "pybemt_optimized_ehang_advanced.yaml")

    # chord[] is nondimensional multiplier of c_ref in your propeller class
    c_ref_override_m: float = 0.15
    force_airfoil_id: int | None = 2412

    # --- Optimization Settings ---
    de_maxiter: int = 10
    de_popsize: int = 10
    stage2_maxiter: int = 600  # increase vs 300 to avoid MAXFUN stop too early

    # --- Solver Settings ---
    FAST_COUPLING_ITERS: int = 5
    STRICT_COUPLING_ITERS: int = 25
    coupling_tol_fast: float = 3e-3
    coupling_tol_strict: float = 1e-3

    # --- Penalty weights (Stage 1) ---
    W_THRUST_SHORTFALL: float = 5000.0
    W_MACH: float = 2000.0

    # --- Regularization weights (keeps “extra” vars from bound-hugging) ---
    W_RATIO_REG: float = 0.0     # set e.g. 50..300 if you want ratio closer to 1.0
    W_S2_REG: float = 80.0       # keeps s2 near 1.0 (tune 20..200)

    # --- Bounds for ratio and s2 ---
    ratio_lb: float = 0.8
    ratio_ub: float = 1.2
    s2_lb: float = 0.90
    s2_ub: float = 1.10

    # --- Twist shape exponent (washout law) ---
    twist_exp: float = 1.2


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
class EvalCache:
    """Cache to avoid re-running BEMT for objective+constraint at identical x."""
    def __init__(self, atol: float = 1e-9, verbose: bool = False):
        self.atol = float(atol)
        self.verbose = bool(verbose)
        self.x_last = None
        self.result = None
        self.n_eval = 0

    def get(self, x, base_rot1, base_rot2, fl, V_inf, r_R, cfg: Config,
            iters: int, W: float, coupling_tol: float):
        x = np.asarray(x, dtype=float)
        if self.x_last is not None and np.allclose(x, self.x_last, rtol=0.0, atol=self.atol):
            return self.result

        self.n_eval += 1
        t0 = time.time()
        if self.verbose:
            print(f"\n[EVAL {self.n_eval:04d}] BEMT start (iters={iters})", flush=True)

        self.result = evaluate_system(
            x, base_rot1, base_rot2, fl, float(V_inf), r_R, cfg,
            max_coupling_iter=int(iters),
            W=float(W),
            coupling_tol=float(coupling_tol),
        )
        self.x_last = x.copy()

        if self.verbose:
            dt = time.time() - t0
            totals, _, _, _, P, r1, r2 = self.result
            T = float(totals["T_total"])
            Rmax = max(float(r1.radius), float(r2.radius))
            Mtip = float(x[8]) * Rmax / float(fl.a)
            print(f"[EVAL {self.n_eval:04d}] done {dt:.1f}s | T={T:.2f} N | P={P:.1f} W | ratio={x[9]:.3f} | s2={x[10]:.3f} | Mtip={Mtip:.3f}",
                  flush=True)
        return self.result


def radial_grid(n_stations: int, r_root_norm: float) -> np.ndarray:
    return np.linspace(float(r_root_norm), 1.0, int(n_stations))


def compute_radius_pair_from_ratio(W: float, disk_loading: float, n_propulsors: int, ratio: float):
    """
    Total coaxial disk area per propulsor: A_total = (W/N_prop) / DL
    Model coaxial pair: A_total = pi*(R1^2 + R2^2), with R1 = ratio*R2
    """
    A_total = (W / float(n_propulsors)) / float(disk_loading)
    R2 = np.sqrt(A_total / (np.pi * (ratio**2 + 1.0)))
    R1 = ratio * R2
    return float(R1), float(R2), float(A_total)


def override_scale_in_memory(prop, R_new: float, c_ref_new: float, nblades_new: int):
    prop.radius = float(R_new)
    prop.c_ref = float(c_ref_new)
    prop.nblades = int(nblades_new)
    return prop


# -----------------------------------------------------------------------------
# Shape-by-construction chord & twist models
# -----------------------------------------------------------------------------
def chord_hump_taper(r_R: np.ndarray, c_root_nd: float, c_peak_nd: float, r_peak: float, c_tip_nd: float) -> np.ndarray:
    """
    Chord model that always produces a single hump near r_peak and tapers to the tip.
    Returns chord as nondimensional multiplier of c_ref.
    """
    r = np.asarray(r_R, dtype=float)

    rpk = float(np.clip(r_peak, r[0] + 1e-6, 1.0))
    c0 = float(max(c_root_nd, 1e-4))
    ct = float(max(c_tip_nd,  1e-4))
    cp = float(max(c_peak_nd, max(c0, ct) + 1e-6))  # enforce peak above root & tip

    w = 0.18  # hump width
    bump = np.exp(-((r - rpk) / w) ** 2)
    root_blend = (1.0 - r) ** 1.5

    c = ct + (cp - ct) * bump + (c0 - ct) * root_blend
    return np.maximum(c, 1e-4)


def washout_pitch(r_R: np.ndarray, beta_root_deg: float, d_beta_deg: float, expn: float) -> np.ndarray:
    """
    Twist model always decreasing with radius (washout):
      beta(r) = beta_root - d_beta * t^expn  with d_beta >= 0
    """
    r = np.asarray(r_R, dtype=float)
    t = (r - r[0]) / max(r[-1] - r[0], 1e-12)
    d = max(float(d_beta_deg), 0.0)
    return float(beta_root_deg) - d * (t ** float(expn))


# -----------------------------------------------------------------------------
# Build rotors from x
# -----------------------------------------------------------------------------
def build_rotors_from_x(x, base_rot1, base_rot2, r_R, cfg: Config, W: float):
    x = np.asarray(x, dtype=float)

    # Chord parameters (upper rotor base shape)
    c_root_nd = float(x[0])
    c_peak_nd = float(x[1])
    r_peak    = float(x[2])
    c_tip_nd  = float(x[3])

    # Twist parameters (washout)
    beta1_root = float(x[4])
    d_beta1    = float(x[5])
    beta2_root = float(x[6])
    d_beta2    = float(x[7])

    omega = float(x[8])
    ratio = float(x[9])

    # New: lower rotor chord scaling
    s2 = float(x[10])
    s2 = float(np.clip(s2, cfg.s2_lb, cfg.s2_ub))

    R1, R2, _ = compute_radius_pair_from_ratio(W, cfg.disk_loading, cfg.n_propulsors, ratio)

    chord_nd = chord_hump_taper(r_R, c_root_nd, c_peak_nd, r_peak, c_tip_nd)
    beta1 = washout_pitch(r_R, beta1_root, d_beta1, cfg.twist_exp)
    beta2 = washout_pitch(r_R, beta2_root, d_beta2, cfg.twist_exp)

    r1 = copy.deepcopy(base_rot1)
    r2 = copy.deepcopy(base_rot2)

    r1 = override_scale_in_memory(r1, R1, cfg.c_ref_override_m, cfg.n_blades)
    r2 = override_scale_in_memory(r2, R2, cfg.c_ref_override_m, cfg.n_blades)

    r1.r_R = np.array(r_R, dtype=float)
    r2.r_R = np.array(r_R, dtype=float)

    # Upper chord base, lower chord scaled
    r1.chord = np.array(chord_nd, dtype=float)
    r2.chord = np.array(chord_nd * s2, dtype=float)

    r1.pitch = np.array(beta1, dtype=float)
    r2.pitch = np.array(beta2, dtype=float)

    if cfg.force_airfoil_id is not None:
        r1.airfoil = np.full_like(r1.r_R, int(cfg.force_airfoil_id), dtype=int)
        r2.airfoil = np.full_like(r2.r_R, int(cfg.force_airfoil_id), dtype=int)

    r1.omega = omega
    r2.omega = omega
    return r1, r2


# -----------------------------------------------------------------------------
# System evaluation
# -----------------------------------------------------------------------------
def evaluate_system(x, base_rot1, base_rot2, fl, V_inf, r_R, cfg: Config,
                    max_coupling_iter: int, W: float, coupling_tol: float):
    r1, r2 = build_rotors_from_x(x, base_rot1, base_rot2, r_R, cfg, W=W)

    totals, out1, out2, coupling = bemt_coaxial.coaxial_bemt_fixed(
        r1, r2, fl,
        V_inf=float(V_inf),
        omega1=float(r1.omega),
        omega2=float(r2.omega),
        r_R=r_R,
        max_coupling_iter=int(max_coupling_iter),
        coupling_tol=float(coupling_tol),
        trim_rotor2=False,
        alpha_target_deg=3.0,
    )

    omega = float(np.asarray(x)[8])
    P_shaft = float(totals["Q_total"] * omega)
    return totals, out1, out2, coupling, P_shaft, r1, r2


# -----------------------------------------------------------------------------
# Objectives & Constraints
# -----------------------------------------------------------------------------
def objective_stage1_hinge(x, base_rot1, base_rot2, fl, V_inf, r_R, cfg: Config,
                          T_target: float, W: float, eval_counter=None, print_every: int = 80):
    if eval_counter is not None:
        eval_counter[0] += 1
        if (eval_counter[0] % int(print_every)) == 0:
            print(".", end="", flush=True)

    totals, _, _, _, P, r1, r2 = evaluate_system(
        x, base_rot1, base_rot2, fl, V_inf, r_R, cfg,
        max_coupling_iter=cfg.FAST_COUPLING_ITERS,
        W=W,
        coupling_tol=cfg.coupling_tol_fast,
    )

    if (not np.isfinite(P)) or P <= 1.0:
        return 1e12

    T = float(totals["T_total"])

    # Thrust shortfall hinge penalty (only if T < target)
    shortfall = max(0.0, (T_target - T) / max(T_target, 1e-9))
    penalty_thrust = cfg.W_THRUST_SHORTFALL * (shortfall ** 2) * P

    # Mach hinge penalty
    R_max = max(float(r1.radius), float(r2.radius))
    M_tip = float(np.asarray(x)[8]) * R_max / float(fl.a)
    margin = cfg.M_tip_max - M_tip
    violation = max(0.0, -margin)
    penalty_mach = cfg.W_MACH * (violation ** 2) * P

    # Regularization: keep ratio near 1 (optional) and s2 near 1 (recommended)
    ratio = float(np.asarray(x)[9])
    s2 = float(np.asarray(x)[10])
    penalty_reg = 0.0
    if cfg.W_RATIO_REG > 0.0:
        penalty_reg += cfg.W_RATIO_REG * ((ratio - 1.0) ** 2) * P
    if cfg.W_S2_REG > 0.0:
        penalty_reg += cfg.W_S2_REG * ((s2 - 1.0) ** 2) * P

    return float(P + penalty_thrust + penalty_mach + penalty_reg)


def constr_thrust_min_cached(x, cache: EvalCache, base_rot1, base_rot2, fl, V_inf, r_R, cfg: Config,
                             T_target: float, W: float):
    totals, _, _, _, _, _, _ = cache.get(
        x, base_rot1, base_rot2, fl, V_inf, r_R, cfg,
        iters=cfg.FAST_COUPLING_ITERS,
        W=W,
        coupling_tol=cfg.coupling_tol_fast,
    )
    return float(totals["T_total"] - T_target)  # >= 0


def constr_mach_max_geom(x, base_rot1, base_rot2, fl, V_inf, r_R, cfg: Config, W: float):
    r1, r2 = build_rotors_from_x(x, base_rot1, base_rot2, r_R, cfg, W=W)
    R_max = max(float(r1.radius), float(r2.radius))
    M_tip = float(np.asarray(x)[8]) * R_max / float(fl.a)
    return float(cfg.M_tip_max - M_tip)  # >= 0


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_results(x_opt, base_rot1, base_rot2, fl, r_R, cfg: Config, T_target: float, W: float):
    totals, out1, out2, coupling, P_shaft, r1, r2 = evaluate_system(
        x_opt, base_rot1, base_rot2, fl, cfg.V_design, r_R, cfg,
        max_coupling_iter=cfg.STRICT_COUPLING_ITERS,
        W=W,
        coupling_tol=cfg.coupling_tol_strict,
    )

    fig, axs = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    axs[0].plot(r_R, r1.pitch, label="Upper Rotor Pitch [deg]")
    axs[0].plot(r_R, r2.pitch, label="Lower Rotor Pitch [deg]")
    axs[0].set_ylabel("Pitch [deg]")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(r_R, np.asarray(r1.chord) * float(r1.c_ref), label="Upper chord [m]")
    axs[1].plot(r_R, np.asarray(r2.chord) * float(r2.c_ref), label="Lower chord [m]", linestyle="--")
    axs[1].set_xlabel("r/R [-]")
    axs[1].set_ylabel("Chord [m]")
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.show()

    # Power sweep (climb domain only)
    print("\nGenerating Power Sweep (V >= 0 only)...")
    v_start = 0.0
    v_end = cfg.V_design + 10.0
    speeds = np.arange(v_start, v_end + 0.1, 1.0)

    P_list, T_list = [], []
    for V in speeds:
        res = bemt_coaxial.coaxial_bemt_fixed(
            r1, r2, fl,
            float(V),
            float(r1.omega), float(r2.omega),
            r_R,
            max_coupling_iter=cfg.FAST_COUPLING_ITERS,
            coupling_tol=cfg.coupling_tol_fast,
            trim_rotor2=False
        )
        P_val = float(res[0]["Q_total"]) * float(r1.omega)
        P_list.append(P_val)
        T_list.append(float(res[0]["T_total"]))

    plt.figure(figsize=(9, 5))
    plt.plot(speeds, P_list, marker="o", label="Shaft Power [W]")
    plt.axvline(cfg.V_design, linestyle="--", color="k", label="Design V")
    plt.xlabel("Climb Speed [m/s]")
    plt.ylabel("Power [W]")
    plt.title("Performance at Constant Optimized RPM (Climb Speeds)")
    plt.grid(True)
    plt.legend()
    plt.show()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    # IMPORTANT: because this is a NEW parameterization (11 vars), old checkpoints are incompatible.
    # Recommend: delete checkpoint_opt0.npy at least once.
    SKIP_STAGE_1 = False  # run DE at least once for this new formulation

    cfg = Config()
    base_prop = propeller.Propeller.load_from_yaml(cfg.yaml_path)

    W = float(cfg.mass_kg * cfg.g)
    T_target_system = W / float(cfg.n_propulsors)

    fl = fluid.Fluid(cfg.altitude_m)

    base_rot1 = copy.deepcopy(base_prop)
    base_rot2 = copy.deepcopy(base_prop)

    r_R = radial_grid(cfg.n_stations, cfg.r_root_norm)

    # Mach-safe omega_max for worst-case Rmax over ratio bounds
    _, _, A_total = compute_radius_pair_from_ratio(W, cfg.disk_loading, cfg.n_propulsors, ratio=cfg.ratio_ub)
    R2_worst = np.sqrt(A_total / (np.pi * (cfg.ratio_ub**2 + 1.0)))
    R1_worst = cfg.ratio_ub * R2_worst
    Rmax_worst = max(R1_worst, R2_worst)
    omega_max = cfg.M_tip_max * float(fl.a) / float(Rmax_worst)

    print(f"Target_Area(total coaxial): {A_total:.4f} m^2")
    print(f"Worst-case Rmax (ratio={cfg.ratio_ub:.2f}): {Rmax_worst:.3f} m")
    print(f"omega_max (Mach-safe): {omega_max:.3f} rad/s")

    # Bounds for 11 variables
    bounds = Bounds(
        # [c_root_nd, c_peak_nd, r_peak, c_tip_nd,
        #  beta1_root, d_beta1, beta2_root, d_beta2,
        #  omega, ratio, s2]
        [0.6,  1.0,  0.25, 0.4,
         5.0,  0.0,  5.0,  0.0,
         0.2*omega_max, cfg.ratio_lb, cfg.s2_lb],

        [3.0,  4.0,  0.55, 2.5,
         50.0, 35.0, 50.0, 35.0,
         omega_max,    cfg.ratio_ub, cfg.s2_ub]
    )

    checkpoint_file = "checkpoint_opt0.npy"

    # Sensible initial guess for Option B (11 vars)
    x0 = np.array([
        1.0,    # c_root_nd
        2.3,    # c_peak_nd
        0.40,   # r_peak
        0.9,    # c_tip_nd
        25.0,   # beta1_root
        15.0,   # d_beta1 (washout)
        22.0,   # beta2_root
        12.0,   # d_beta2
        0.7*omega_max,
        1.0,    # ratio
        1.0,    # s2
    ], dtype=float)

    # Only load checkpoint if it matches 11 vars (and you know it was produced by THIS code)
    if os.path.exists(checkpoint_file):
        try:
            xchk = np.load(checkpoint_file)
            if np.asarray(xchk).shape == x0.shape:
                print(f"[INFO] Loaded checkpoint: {checkpoint_file}")
                x0 = np.asarray(xchk, dtype=float)
            else:
                print(f"[INFO] Ignoring checkpoint (shape mismatch): {checkpoint_file}")
        except Exception:
            print("[INFO] Failed to load checkpoint safely; using x0.")

    # Quick sanity check at x0
    print("\n[CHECK] Evaluating system at x0 (fast) ...")
    totals0, *_ = evaluate_system(
        x0, base_rot1, base_rot2, fl, cfg.V_design, r_R, cfg,
        max_coupling_iter=cfg.FAST_COUPLING_ITERS,
        W=W,
        coupling_tol=cfg.coupling_tol_fast,
    )
    print(f"[CHECK] T_total={float(totals0['T_total']):.3f} N | Target={T_target_system:.3f} N")

    # --- STAGE 1: Differential Evolution ---
    if SKIP_STAGE_1 and os.path.exists(checkpoint_file):
        print("\n=== SKIPPING STAGE 1 (Using Checkpoint) ===")
        x1 = x0
    else:
        print("\n=== STAGE 1: Differential Evolution (Hinge Objective) ===")
        eval_counter = [0]

        def cb_save(xk, convergence=None):
            np.save(checkpoint_file, np.asarray(xk, dtype=float))
            print(" [SAVED]", end="", flush=True)

        res1 = differential_evolution(
            func=lambda x: objective_stage1_hinge(
                x, base_rot1, base_rot2, fl, cfg.V_design, r_R, cfg,
                T_target=T_target_system,
                W=W,
                eval_counter=eval_counter,
                print_every=80
            ),
            bounds=bounds,
            strategy="best1bin",
            maxiter=cfg.de_maxiter,
            popsize=cfg.de_popsize,
            tol=0.01,
            workers=1,
            updating="deferred",
            callback=cb_save,
            disp=True,
            polish=True
        )
        x1 = np.asarray(res1.x, dtype=float)

    # --- STAGE 2: COBYLA (Derivative-Free) + caching ---
    print("\n=== STAGE 2: COBYLA (Derivative-Free Polish) ===")
    cache = EvalCache(atol=1e-9, verbose=False)

    def obj2(x):
        totals, _, _, _, P, _, _ = cache.get(
            x, base_rot1, base_rot2, fl, cfg.V_design, r_R, cfg,
            iters=cfg.FAST_COUPLING_ITERS,
            W=W,
            coupling_tol=cfg.coupling_tol_fast,
        )
        if (not np.isfinite(P)) or P <= 1.0:
            return 1e12

        # soft stabilization penalty (constraints enforce feasibility)
        T = float(totals["T_total"])
        shortfall = max(0.0, (T_target_system - T) / max(T_target_system, 1e-9))

        # very light regularization to avoid s2 and ratio bound-hugging in the local stage
        ratio = float(np.asarray(x)[9])
        s2 = float(np.asarray(x)[10])
        reg = 0.0
        if cfg.W_RATIO_REG > 0.0:
            reg += cfg.W_RATIO_REG * ((ratio - 1.0) ** 2)
        if cfg.W_S2_REG > 0.0:
            reg += cfg.W_S2_REG * ((s2 - 1.0) ** 2)

        return float(P * (1.0 + 0.05 * shortfall) + reg)

    lb = bounds.lb
    ub = bounds.ub

    cons = []
    cons.append({"type": "ineq", "fun": lambda x: constr_thrust_min_cached(
        x, cache, base_rot1, base_rot2, fl, cfg.V_design, r_R, cfg, T_target_system, W
    )})
    cons.append({"type": "ineq", "fun": lambda x: constr_mach_max_geom(
        x, base_rot1, base_rot2, fl, cfg.V_design, r_R, cfg, W
    )})

    # bounds-as-constraints for COBYLA
    for i in range(len(x0)):
        cons.append({"type": "ineq", "fun": lambda x, i=i: float(x[i] - lb[i])})
        cons.append({"type": "ineq", "fun": lambda x, i=i: float(ub[i] - x[i])})

    res2 = minimize(
        fun=obj2,
        x0=x1,
        method="COBYLA",
        constraints=cons,
        options={"maxiter": cfg.stage2_maxiter, "rhobeg": 0.08, "disp": True}
    )

    x_opt = np.asarray(res2.x, dtype=float)
    print(f"\nStage 2 success: {res2.success}")
    print(f"Stage 2 message: {res2.message}")

    # --- FINAL STRICT EVALUATION ---
    print("\n=== FINAL STRICT EVALUATION (High Fidelity) ===")
    totals, _, _, _, P_shaft, r1, r2 = evaluate_system(
        x_opt, base_rot1, base_rot2, fl, cfg.V_design, r_R, cfg,
        max_coupling_iter=cfg.STRICT_COUPLING_ITERS,
        W=W,
        coupling_tol=cfg.coupling_tol_strict,
    )

    Rmax = max(float(r1.radius), float(r2.radius))
    M_tip = float(x_opt[8]) * Rmax / float(fl.a)

    print("\n[FINAL RESULTS]")
    print(f"Thrust : {float(totals['T_total']):.2f} N (Target: {T_target_system:.2f} N)")
    print(f"Power  : {P_shaft:.2f} W")
    print(f"omega  : {x_opt[8]:.4f} rad/s")
    print(f"ratio  : {x_opt[9]:.4f}")
    print(f"s2     : {x_opt[10]:.4f} (lower chord scale)")
    print(f"R_upper: {float(r1.radius):.4f} m")
    print(f"R_lower: {float(r2.radius):.4f} m")
    print(f"M_tip  : {M_tip:.4f} (limit {cfg.M_tip_max:.4f})")

    # Save BOTH rotors
    base, ext = os.path.splitext(cfg.out_yaml_path)
    out1_path = base + "_upper" + ext
    out2_path = base + "_lower" + ext
    r1.save_to_yaml(out1_path)
    r2.save_to_yaml(out2_path)
    print(f"\nSaved Upper Rotor to: {out1_path}")
    print(f"Saved Lower Rotor to: {out2_path}")

    plot_results(x_opt, base_rot1, base_rot2, fl, r_R, cfg, T_target_system, W=W)


if __name__ == "__main__":
    main()
