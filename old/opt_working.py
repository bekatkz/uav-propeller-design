# opt0.py
"""
COAXIAL PROPELLER OPTIMIZATION (Robust COBYLA + Caching)
- Stage 1: Differential Evolution (Global Search) with Hinge Penalty (T >= target)
- Stage 2: COBYLA (Derivative-Free Local Search) + EvalCache to avoid duplicate BEMT calls
- FAST_COUPLING_ITERS used during optimization; STRICT_COUPLING_ITERS used for final report

Key fixes included:
(A) omega_max bound is made Mach-safe for the *worst-case* R_max over the allowed ratio range
(B) W (weight) is passed explicitly through build/evaluate to prevent hidden inconsistencies
(C) Saves both Upper and Lower rotor YAMLs
"""

from dataclasses import dataclass
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, differential_evolution
from scipy.interpolate import CubicSpline

# Ensure these modules are in the same folder
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
    n_rotors: int = 16

    # --- Geometry Definition ---
    n_stations: int = 30
    r_root_norm: float = 0.10

    # --- File Paths ---
    yaml_path: str = os.path.join("data", "pybemt_tmotor28.yaml")
    out_yaml_path: str = os.path.join("data", "pybemt_optimized_ehang_advanced.yaml")

    c_ref_override_m: float = 0.15
    force_airfoil_id: int | None = 2412

    # --- Optimization Settings ---
    de_maxiter: int = 10
    de_popsize: int = 10
    stage2_maxiter: int = 300

    # --- TUNING CONSTANTS ---
    FAST_COUPLING_ITERS: int = 5
    STRICT_COUPLING_ITERS: int = 25
    W_THRUST_SHORTFALL: float = 5000.0
    W_MACH: float = 2000.0

    # --- Design variable bounds (ratio bounds used also for omega_max safe bound) ---
    ratio_lb: float = 0.8
    ratio_ub: float = 1.2


# -----------------------------------------------------------------------------
# Helpers & Caching
# -----------------------------------------------------------------------------
class EvalCache:
    """Cache to prevent re-evaluating BEMT for Objective AND Constraints at same x."""
    def __init__(self, atol: float = 1e-9):
        self.x_last = None
        self.result = None
        self.atol = float(atol)

    def get(self, x, base_rot1, base_rot2, fl, V_inf, r_R, cfg, iters, W):
        x = np.asarray(x, dtype=float)
        if self.x_last is not None and np.allclose(x, self.x_last, rtol=0.0, atol=self.atol):
            return self.result
        self.result = evaluate_system(
            x, base_rot1, base_rot2, fl, V_inf, r_R, cfg,
            max_coupling_iter=int(iters),
            W=float(W)
        )
        self.x_last = x.copy()
        return self.result


def radial_grid(n_stations: int, r_root_norm: float) -> np.ndarray:
    return np.linspace(float(r_root_norm), 1.0, int(n_stations))


def chord_cubic_spline(r_R: np.ndarray, c_ctrl: np.ndarray) -> np.ndarray:
    r_pts = np.array([float(r_R[0]), 0.45, 0.75, 1.0], dtype=float)
    c_pts = np.array(c_ctrl, dtype=float)
    cs = CubicSpline(r_pts, c_pts, bc_type="natural")
    c = cs(r_R)
    return np.maximum(c, 1e-4)


def linear_pitch(r_R: np.ndarray, beta_root_deg: float, beta_tip_deg: float) -> np.ndarray:
    r0 = float(r_R[0])
    r1 = float(r_R[-1])
    t = (r_R - r0) / max(r1 - r0, 1e-12)
    return float(beta_root_deg) + (float(beta_tip_deg) - float(beta_root_deg)) * t


def compute_radius_pair_from_ratio(W: float, disk_loading: float, n_propulsors: int, ratio: float):
    """
    Total coaxial disk area per propulsor: A_total = (W/N_prop) / DL
    Coaxial pair modeled as A_total = pi*(R1^2 + R2^2), with R1 = ratio*R2
    """
    Target_Area = (W / float(n_propulsors)) / float(disk_loading)
    R2 = np.sqrt(Target_Area / (np.pi * (ratio**2 + 1.0)))
    R1 = ratio * R2
    return float(R1), float(R2)


def override_scale_in_memory(prop, R_new: float, c_ref_new: float, nblades_new: int):
    prop.radius = float(R_new)
    prop.c_ref = float(c_ref_new)
    prop.nblades = int(nblades_new)
    return prop


def make_checkpoint_callback(filename):
    def callback(xk, convergence=None):
        np.save(filename, xk)
        print(" [SAVED]", end="")
    return callback


# -----------------------------------------------------------------------------
# Build rotors
# -----------------------------------------------------------------------------
def build_rotors_from_x(x, base_rot1, base_rot2, r_R, cfg: Config, W: float):
    c_ctrl = np.asarray(x[0:4], dtype=float)
    b1_root, b1_tip = float(x[4]), float(x[5])
    b2_root, b2_tip = float(x[6]), float(x[7])
    omega = float(x[8])
    ratio = float(x[9])

    R1, R2 = compute_radius_pair_from_ratio(W, cfg.disk_loading, cfg.n_propulsors, ratio)

    chord_dist = chord_cubic_spline(r_R, c_ctrl)
    beta1 = linear_pitch(r_R, b1_root, b1_tip)
    beta2 = linear_pitch(r_R, b2_root, b2_tip)

    r1 = copy.deepcopy(base_rot1)
    r2 = copy.deepcopy(base_rot2)

    r1 = override_scale_in_memory(r1, R1, cfg.c_ref_override_m, cfg.n_blades)
    r2 = override_scale_in_memory(r2, R2, cfg.c_ref_override_m, cfg.n_blades)

    r1.r_R = np.array(r_R, dtype=float)
    r2.r_R = np.array(r_R, dtype=float)
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
# System evaluation
# -----------------------------------------------------------------------------
def evaluate_system(x, base_rot1, base_rot2, fl, V_inf, r_R, cfg: Config, max_coupling_iter: int, W: float):
    r1, r2 = build_rotors_from_x(x, base_rot1, base_rot2, r_R, cfg, W=W)

    totals, out1, out2, coupling = bemt_coaxial.coaxial_bemt_fixed(
        r1, r2, fl,
        V_inf=float(V_inf),
        omega1=float(r1.omega),
        omega2=float(r2.omega),
        r_R=r_R,
        max_coupling_iter=int(max_coupling_iter),
        coupling_tol=1e-3,
        trim_rotor2=False,
        alpha_target_deg=3.0,
    )

    omega = float(x[8])
    P_shaft = float(totals["Q_total"] * omega)  # shaft power for the coaxial system
    return totals, out1, out2, coupling, P_shaft, r1, r2


# -----------------------------------------------------------------------------
# OBJECTIVE (Stage 1 Hinge)
# -----------------------------------------------------------------------------
def objective_stage1_hinge(x, base_rot1, base_rot2, fl, V_inf, r_R, cfg: Config, T_target: float, W: float,
                          eval_counter=None, print_every: int = 50):
    # Optional lightweight progress printing (avoid dot spam)
    if eval_counter is not None:
        eval_counter[0] += 1
        if (eval_counter[0] % int(print_every)) == 0:
            print(".", end="", flush=True)

    totals, _, _, _, P, r1, r2 = evaluate_system(
        x, base_rot1, base_rot2, fl, V_inf, r_R, cfg,
        max_coupling_iter=cfg.FAST_COUPLING_ITERS,
        W=W
    )

    if (not np.isfinite(P)) or P <= 1.0:
        return 1e12

    T = float(totals["T_total"])

    # One-sided thrust penalty: only penalize if below target
    shortfall = max(0.0, (T_target - T) / max(T_target, 1e-9))
    penalty_thrust = cfg.W_THRUST_SHORTFALL * (shortfall ** 2) * P

    # One-sided Mach penalty
    R_max = max(r1.radius, r2.radius)
    M_tip = float(x[8]) * R_max / float(fl.a)
    mach_margin = cfg.M_tip_max - M_tip  # >=0 OK
    violation = max(0.0, -float(mach_margin))
    penalty_mach = cfg.W_MACH * (violation ** 2) * P

    return float(P + penalty_thrust + penalty_mach)


# -----------------------------------------------------------------------------
# Constraints
# -----------------------------------------------------------------------------
def constr_mach_max(x, base_rot1, base_rot2, fl, V_inf, r_R, cfg: Config, W: float):
    # Geometry-only; cheap
    r1, r2 = build_rotors_from_x(x, base_rot1, base_rot2, r_R, cfg, W=W)
    R_max = max(r1.radius, r2.radius)
    M_tip = float(x[8]) * R_max / float(fl.a)
    return float(cfg.M_tip_max - M_tip)  # >= 0


# -----------------------------------------------------------------------------
# PLOTTING
# -----------------------------------------------------------------------------
def plot_results(x_opt, base_rot1, base_rot2, fl, r_R, cfg: Config, T_target: float, W: float):
    totals, out1, out2, coupling, P_shaft, r1, r2 = evaluate_system(
        x_opt, base_rot1, base_rot2, fl, cfg.V_design, r_R, cfg,
        max_coupling_iter=cfg.STRICT_COUPLING_ITERS,
        W=W
    )

    # 1) Geometry
    fig, axs = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    axs[0].plot(r_R, r1.pitch, label="Upper Rotor Pitch [deg]")
    axs[0].plot(r_R, r2.pitch, label="Lower Rotor Pitch [deg]")
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

    # 2) Power sweep at constant optimized omega
    print("\nGenerating Power Sweep...")
    v_start, v_end = cfg.V_design + 10.0, cfg.V_design - 10.0
    speeds = np.arange(v_start, v_end - 0.1, -1.0)
    P_list, T_list = [], []

    for V in speeds:
        res = bemt_coaxial.coaxial_bemt_fixed(
            r1, r2, fl, float(V), float(r1.omega), float(r2.omega), r_R,
            max_coupling_iter=cfg.FAST_COUPLING_ITERS,
            trim_rotor2=False
        )
        P_list.append(float(res[0]["Q_total"]) * float(r1.omega))
        T_list.append(float(res[0]["T_total"]))

    plt.figure(figsize=(9, 5))
    plt.plot(speeds, P_list, marker="o", label="Shaft Power [W]")
    plt.axvline(cfg.V_design, linestyle="--", color="k", label="Design V")
    plt.xlabel("Climb Speed [m/s]")
    plt.ylabel("Power [W]")
    plt.title("Performance at Constant Optimized RPM")
    plt.grid(True)
    plt.legend()
    plt.gca().invert_xaxis()
    plt.show()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    # Set True if you want to skip DE and start from checkpoint
    SKIP_STAGE_1 = True

    cfg = Config()
    base_prop = propeller.Propeller.load_from_yaml(cfg.yaml_path)

    W = float(cfg.mass_kg * cfg.g)
    T_target_system = W / float(cfg.n_propulsors)

    fl = fluid.Fluid(cfg.altitude_m)

    base_rot1 = copy.deepcopy(base_prop)
    base_rot2 = copy.deepcopy(base_prop)
    r_R = radial_grid(cfg.n_stations, cfg.r_root_norm)

    # --- Mach-safe omega_max bound for worst-case Rmax over ratio range ---
    Target_Area = (W / cfg.n_propulsors) / cfg.disk_loading  # total coaxial disk area per propulsor
    ratio_lb = float(cfg.ratio_lb)
    ratio_ub = float(cfg.ratio_ub)

    # Worst-case largest rotor occurs at ratio extremes (use ratio_ub here)
    R2_worst = np.sqrt(Target_Area / (np.pi * (ratio_ub**2 + 1.0)))
    R1_worst = ratio_ub * R2_worst
    Rmax_worst = max(R1_worst, R2_worst)

    omega_max = cfg.M_tip_max * float(fl.a) / float(Rmax_worst)

    print(f"Target_Area(total coaxial): {Target_Area:.4f} m^2")
    print(f"Worst-case Rmax (ratio={ratio_ub:.2f}): {Rmax_worst:.3f} m")
    print(f"omega_max (Mach-safe): {omega_max:.3f} rad/s")

    bounds = Bounds(
        [0.5]*4 + [-5.0]*4 + [0.2*omega_max] + [ratio_lb],
        [2.5]*4 + [45.0]*4 + [omega_max]     + [ratio_ub]
    )

    # Initial guess
    checkpoint_file = "checkpoint_opt0.npy"
    x0 = np.array([1.0]*4 + [25, 10, 22, 9] + [0.7*omega_max] + [1.0], dtype=float)

    if os.path.exists(checkpoint_file):
        print(f"[INFO] Loaded checkpoint: {checkpoint_file}")
        x0 = np.load(checkpoint_file)

    # --- STAGE 1: Differential Evolution ---
    if SKIP_STAGE_1 and os.path.exists(checkpoint_file):
        print("\n=== SKIPPING STAGE 1 (Using Checkpoint) ===")
        x1 = x0
    else:
        print("\n=== STAGE 1: Differential Evolution (Hinge Objective) ===")
        eval_counter = [0]
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
            workers=-1,
            updating="deferred",
            callback=make_checkpoint_callback(checkpoint_file),
            disp=True
        )
        x1 = res1.x

    # --- STAGE 2: COBYLA with EVALUATION CACHE ---
    print("\n=== STAGE 2: COBYLA (Derivative-Free Polish) ===")
    cache = EvalCache(atol=1e-9)

    def obj2_cached(x):
        totals, _, _, _, P, _, _ = cache.get(
            x, base_rot1, base_rot2, fl, cfg.V_design, r_R, cfg,
            iters=cfg.FAST_COUPLING_ITERS,
            W=W
        )
        if (not np.isfinite(P)) or P <= 1.0:
            return 1e12

        # soft stabilization penalty (does NOT enforce feasibility; constraints do)
        T = float(totals["T_total"])
        shortfall = max(0.0, (T_target_system - T) / max(T_target_system, 1e-9))
        return float(P * (1.0 + 0.05 * shortfall))

    def constr_thrust_cached(x):
        totals, _, _, _, _, _, _ = cache.get(
            x, base_rot1, base_rot2, fl, cfg.V_design, r_R, cfg,
            iters=cfg.FAST_COUPLING_ITERS,
            W=W
        )
        return float(totals["T_total"] - T_target_system)  # >= 0

    lb = bounds.lb
    ub = bounds.ub

    cons_cobyla = []
    cons_cobyla.append({"type": "ineq", "fun": constr_thrust_cached})
    cons_cobyla.append({"type": "ineq", "fun": lambda x: constr_mach_max(x, base_rot1, base_rot2, fl, cfg.V_design, r_R, cfg, W)})

    # Bound constraints (COBYLA doesn't support Bounds directly)
    for i in range(len(x0)):
        cons_cobyla.append({"type": "ineq", "fun": lambda x, i=i: float(x[i] - lb[i])})
        cons_cobyla.append({"type": "ineq", "fun": lambda x, i=i: float(ub[i] - x[i])})

    res2 = minimize(
        fun=obj2_cached,
        x0=x1,
        method="COBYLA",
        constraints=cons_cobyla,
        options={"maxiter": cfg.stage2_maxiter, "rhobeg": 0.1, "disp": True}
    )

    x_opt = np.asarray(res2.x, dtype=float)
    print(f"\nStage 2 success: {res2.success}")
    print(f"Stage 2 message: {res2.message}")

    # --- FINAL STRICT EVALUATION ---
    print("\n=== FINAL STRICT EVALUATION (High Fidelity) ===")
    totals, _, _, _, P_shaft, r1, r2 = evaluate_system(
        x_opt, base_rot1, base_rot2, fl, cfg.V_design, r_R, cfg,
        max_coupling_iter=cfg.STRICT_COUPLING_ITERS,
        W=W
    )

    print("\n[FINAL RESULTS]")
    print(f"Thrust : {float(totals['T_total']):.2f} N (Target: {T_target_system:.2f} N)")
    print(f"Power  : {P_shaft:.2f} W")
    print(f"omega  : {x_opt[8]:.4f} rad/s")
    print(f"ratio  : {x_opt[9]:.4f}")
    print(f"R_upper: {r1.radius:.4f} m")
    print(f"R_lower: {r2.radius:.4f} m")
    print(f"M_tip  : {(x_opt[8]*max(r1.radius, r2.radius)/fl.a):.4f} (limit {cfg.M_tip_max:.4f})")

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
