# opt0.py
"""
COAXIAL PROPELLER OPTIMIZATION (Advanced - Checkpointing Version)
- Keeps 'def main()' for clean code (no global variables).
- Auto-saves 'checkpoint_opt0.npy' after every generation.
- Allows resuming/skipping Stage 1 if a checkpoint is found.
"""

from dataclasses import dataclass
import os
import copy
import time
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
    stage2_maxiter: int = 250
    ftol2: float = 1e-7
    sweep_span: float = 10.0
    sweep_step: float = 1.0


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
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
    T_pair = W / n_propulsors
    Target_Area = (W / float(n_propulsors)) / float(disk_loading)
    R2 = np.sqrt(Target_Area / (np.pi * (ratio**2 + 1.0)))
    R1 = ratio * R2
    return float(R1), float(R2)

def override_scale_in_memory(prop, R_new: float, c_ref_new: float, nblades_new: int):
    prop.radius = float(R_new)
    prop.c_ref = float(c_ref_new)
    prop.nblades = int(nblades_new)
    return prop

# -----------------------------------------------------------------------------
# CHECKPOINT CALLBACK (The Safety Net)
# -----------------------------------------------------------------------------
def make_checkpoint_callback(filename):
    """Creates a callback function that saves x to a file every generation."""
    def callback(xk, convergence=None):
        np.save(filename, xk)
        # We print a small marker so you know it saved
        print(f" [SAVED]", end="")
    return callback

# -----------------------------------------------------------------------------
# Build rotors
# -----------------------------------------------------------------------------
def build_rotors_from_x(x, base_rot1, base_rot2, r_R, cfg: Config):
    c_ctrl = np.asarray(x[0:4], dtype=float)
    b1_root, b1_tip = float(x[4]), float(x[5])
    b2_root, b2_tip = float(x[6]), float(x[7])
    omega = float(x[8])
    ratio = float(x[9])

    R1, R2 = compute_radius_pair_from_ratio(cfg.mass_kg * cfg.g, cfg.disk_loading, cfg.n_propulsors, ratio)

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
    else:
        if hasattr(r1, "airfoil") and len(r1.airfoil) > 0:
            r1.airfoil = np.full_like(r1.r_R, int(r1.airfoil[0]), dtype=int)
            r2.airfoil = np.full_like(r2.r_R, int(r2.airfoil[0]), dtype=int)

    r1.omega = omega
    r2.omega = omega
    return r1, r2

# -----------------------------------------------------------------------------
# System evaluation
# -----------------------------------------------------------------------------
def evaluate_system(x, base_rot1, base_rot2, fl, V_inf, r_R, cfg: Config, max_coupling_iter: int = 25):
    r1, r2 = build_rotors_from_x(x, base_rot1, base_rot2, r_R, cfg)
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
    P_shaft = float(totals["Q_total"] * omega)
    return totals, out1, out2, coupling, P_shaft, r1, r2

def _assert_finite_eval(tag, totals, P):
    if not np.isfinite(P): raise ValueError(f"[{tag}] Power is not finite: {P}")
    for k in ("T_total", "T1", "T2", "Q_total", "Q1", "Q2"):
        if k in totals and (not np.isfinite(totals[k])):
            raise ValueError(f"[{tag}] totals[{k}] is not finite: {totals[k]}")

# -----------------------------------------------------------------------------
# Objective & constraints
# -----------------------------------------------------------------------------
def objective_power(x, base_rot1, base_rot2, fl, V_inf, r_R, cfg: Config):
    totals, _, _, _, P, _, _ = evaluate_system(x, base_rot1, base_rot2, fl, V_inf, r_R, cfg)
    if (not np.isfinite(P)) or P <= 0: return 1e12
    if totals["T1"] <= 0 or totals["T2"] <= 0: return float(P + 1e8)
    return float(P)

def objective_stage1_penalty(x, base_rot1, base_rot2, fl, V_inf, r_R, cfg: Config, T_target: float):
    print(".", end="", flush=True) 
    totals, _, _, _, P, _, _ = evaluate_system(
        x, base_rot1, base_rot2, fl, V_inf, r_R, cfg, max_coupling_iter=5
    )
    if (not np.isfinite(P)) or P <= 1.0: return 1e15 
    T = float(totals["T_total"])
    thrust_err = (T - T_target) / T_target 
    penalty_thrust = 500.0 * (thrust_err ** 2) * P 
    
    r1_temp, r2_temp = build_rotors_from_x(x, base_rot1, base_rot2, r_R, cfg)
    R_max = max(r1_temp.radius, r2_temp.radius)
    omega = x[8]
    M_tip = omega * R_max / fl.a
    penalty_mach = 0.0
    if M_tip > cfg.M_tip_max: penalty_mach = 1e6 * (M_tip - cfg.M_tip_max)
    return float(P + penalty_thrust + penalty_mach)

def thrust_eq_constraint(x, base_rot1, base_rot2, fl, V_inf, r_R, cfg: Config, T_target: float):
    totals, _, _, _, _, _, _ = evaluate_system(x, base_rot1, base_rot2, fl, V_inf, r_R, cfg)
    return float(totals["T_total"] - T_target)

def tip_mach_ineq_constraint(x, base_rot1, base_rot2, r_R, cfg, fl, M_tip_max):
    r1, r2 = build_rotors_from_x(x, base_rot1, base_rot2, r_R, cfg)
    R_max = max(r1.radius, r2.radius)
    omega = float(x[8])
    M_tip = omega * float(R_max) / float(fl.a)
    return float(M_tip_max - M_tip)

def trim_omega_for_thrust_capped(x_base, base_rot1, base_rot2, fl, V_inf, r_R, cfg: Config, T_target, omega_min, omega_max, max_bisect_iter=25, n_scan=21, thrust_tol=1e-2):
    x_tmp = np.array(x_base, dtype=float).copy()
    omega0 = float(x_tmp[8])
    om_min = float(max(omega_min, 0.60 * omega0))
    om_max = float(min(omega_max, 1.40 * omega0))
    if om_max <= om_min:
        om_min = float(max(omega_min, 0.80 * omega0))
        om_max = float(min(omega_max, 1.20 * omega0))
    def eval_f(omega):
        x_tmp[8] = float(omega)
        totals, _, _, _, P, _, _ = evaluate_system(x_tmp, base_rot1, base_rot2, fl, V_inf, r_R, cfg)
        f = float(totals["T_total"]) - float(T_target)
        if (not np.isfinite(f)) or (not np.isfinite(P)): return np.nan, totals, np.nan
        return f, totals, float(P)
    f_lo, tot_lo, P_lo = eval_f(om_min)
    f_hi, tot_hi, P_hi = eval_f(om_max)
    if np.isfinite(f_lo) and abs(f_lo) < thrust_tol: return om_min, tot_lo, P_lo
    if np.isfinite(f_hi) and abs(f_hi) < thrust_tol: return om_max, tot_hi, P_hi
    if np.isfinite(f_lo) and np.isfinite(f_hi) and (f_lo * f_hi < 0.0):
        lo, hi = om_min, om_max
        flo = float(f_lo)
        best = None
        for _ in range(int(max_bisect_iter)):
            mid = 0.5 * (lo + hi)
            fmid, tot_mid, P_mid = eval_f(mid)
            if np.isfinite(fmid):
                af = abs(float(fmid))
                if (best is None) or (af < best[0]): best = (af, mid, tot_mid, P_mid)
                if af < thrust_tol: return mid, tot_mid, P_mid
                if flo * float(fmid) < 0.0: hi = mid
                else: lo = mid; flo = float(fmid)
            else: hi = mid
        if best is not None: return best[1], best[2], best[3]
    omegas = np.linspace(om_min, om_max, int(n_scan))
    best = None
    for om in omegas:
        fval, tot, P = eval_f(om)
        if not np.isfinite(fval): continue
        af = abs(float(fval))
        if (best is None) or (af < best[0]):
            best = (af, float(om), tot, float(P))
            if af < thrust_tol: break
    if best is not None: return best[1], best[2], best[3]
    return omega0, tot_lo, P_lo 

def plot_power_sweep_trimmed(x_opt, base_rot1, base_rot2, fl, r_R, cfg: Config, T_target, omega_min, omega_max):
    v_start = cfg.V_design + 10.0
    v_end = cfg.V_design - 10.0
    speeds = np.arange(v_start, v_end - 0.1, -1.0)
    P_shaft_list, P_aero_list, T_list, omega_list = [], [], [], []
    for V in speeds:
        print(f"[SWEEP] V={V:+.1f} m/s -> trimming omega...")
        omega_trim, totals_s, P_s = trim_omega_for_thrust_capped(x_opt, base_rot1, base_rot2, fl, float(V), r_R, cfg, T_target=T_target, omega_min=omega_min, omega_max=omega_max)
        T_s = float(totals_s.get("T_total", np.nan))
        P_shaft_list.append(P_s)
        P_aero_list.append(P_s - T_s * float(V) if (np.isfinite(P_s) and np.isfinite(T_s)) else np.nan)
        T_list.append(T_s)
        omega_list.append(float(omega_trim))
    plt.figure(figsize=(9, 5))
    plt.plot(speeds, P_shaft_list, marker="o", label="P_shaft = QΩ (Total)")
    plt.plot(speeds, P_aero_list, marker="s", label="P_aero (Excl. Climb Power)")
    plt.axvline(cfg.V_design, linestyle="--", color='k', alpha=0.5, label="Design V")
    plt.xlabel("Climb Speed V_inf [m/s]")
    plt.ylabel("Power [W]")
    plt.title("Power vs Climb Speed (Thrust Trimmed)")
    plt.grid(True)
    plt.legend()
    plt.gca().invert_xaxis() 
    plt.show()

def plot_required_distributions(r_R, r1, r2, out1, out2):
    fig, axs = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    axs[0].plot(r_R, r1.pitch, label="Rotor 1 pitch [deg]")
    axs[0].plot(r_R, r2.pitch, label="Rotor 2 pitch [deg]")
    axs[0].set_ylabel("Pitch [deg]")
    axs[0].grid(True)
    axs[0].legend()
    axs[1].plot(r_R, np.asarray(r1.chord) * float(r1.c_ref), label="Chord [m] (shared)")
    axs[1].set_xlabel("r/R [-]")
    axs[1].set_ylabel("Chord [m]")
    axs[1].grid(True)
    axs[1].legend()
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    axs[0].plot(r_R, np.rad2deg(out1["phi"]), label="Rotor 1 phi [deg]")
    axs[0].plot(r_R, np.rad2deg(out2["phi"]), label="Rotor 2 phi [deg]")
    axs[0].set_ylabel("Inflow angle phi [deg]")
    axs[0].grid(True)
    axs[0].legend()
    axs[1].plot(r_R, out1["dT_dr"], label="Rotor 1 dT/dr [N/m]")
    axs[1].plot(r_R, out2["dT_dr"], label="Rotor 2 dT/dr [N/m]")
    axs[1].set_xlabel("r/R [-]")
    axs[1].set_ylabel("Produced thrust dT/dr [N/m]")
    axs[1].grid(True)
    axs[1].legend()
    plt.tight_layout()
    plt.show()

def sanity_checks(cfg: Config, base_rot1, base_rot2, fl, r_R, x0, omega_min, omega_max, T_target):
    print("bemt_coaxial loaded from:", bemt_coaxial.__file__)
    print("\n[CHECK] Evaluating system at x0 ...")
    totals, _, _, coupling, P, _, _ = evaluate_system(x0, base_rot1, base_rot2, fl, cfg.V_design, r_R, cfg)
    _assert_finite_eval("x0", totals, P)
    print(f"[CHECK] OK: T_total={totals['T_total']:.3f} N, P={P:.3f} W")
    r1, r2 = build_rotors_from_x(x0, base_rot1, base_rot2, r_R, cfg)
    R_max = max(r1.radius, r2.radius)
    mach_margin = tip_mach_ineq_constraint(x0, base_rot1, base_rot2, r_R, cfg, fl, cfg.M_tip_max)
    print(f"[CHECK] Tip Mach margin (>=0 is OK): {mach_margin:.6f} (R_max={R_max:.3f}m)")
    if omega_min <= 0 or omega_max <= 0 or omega_max <= omega_min:
        raise ValueError("[CHECK] omega_min/omega_max invalid.")

# -----------------------------------------------------------------------------
# Main (Clean Scope + Checkpointing)
# -----------------------------------------------------------------------------
def main():
    # SET THIS TO TRUE to skip the long wait and jump straight to the final results
    SKIP_STAGE_1 = True  

    cfg = Config()
    base_prop = propeller.Propeller.load_from_yaml(cfg.yaml_path)

    W = cfg.mass_kg * cfg.g
    T_target_system = W / float(cfg.n_propulsors)
    fl = fluid.Fluid(cfg.altitude_m)

    R_est = np.sqrt((W / cfg.n_propulsors) / cfg.disk_loading / (2 * np.pi))
    print(f"Estimated Radius (avg): {R_est:.3f} m")

    base_rot1 = copy.deepcopy(base_prop)
    base_rot2 = copy.deepcopy(base_prop)
    r_R = radial_grid(cfg.n_stations, cfg.r_root_norm)

    omega_max = cfg.M_tip_max * float(fl.a) / R_est
    omega_min = 0.20 * omega_max
    omega0 = 0.70 * omega_max

    c_lb, c_ub = 0.5, 2.5 
    b_lb, b_ub = -5.0, 45.0
    r_lb, r_ub = 0.8, 1.2  

    bounds = Bounds(
        [c_lb, c_lb, c_lb, c_lb, b_lb, b_lb, b_lb, b_lb, omega_min, r_lb],
        [c_ub, c_ub, c_ub, c_ub, b_ub, b_ub, b_ub, b_ub, omega_max, r_ub],
    )

    # --- CHECKPOINT LOGIC ---
    checkpoint_file = "checkpoint_opt0.npy"
    
    # Default initial guess
    x0 = np.array([
        1.0, 1.0, 1.0, 0.8,     # Chord
        25.0, 10.0,             # Pitch1
        22.0, 9.0,              # Pitch2
        omega0,                 # Omega
        1.0                     # Ratio
    ], dtype=float)

    if os.path.exists(checkpoint_file):
        print(f"\n[INFO] Found checkpoint file: {checkpoint_file}")
        try:
            x_loaded = np.load(checkpoint_file)
            print(f"[INFO] Loaded best values from checkpoint: {x_loaded}")
            # Use loaded values as starting point
            x0 = x_loaded
        except:
            print("[WARN] Could not load checkpoint. Starting fresh.")
    else:
        print("[INFO] No checkpoint found. Starting fresh.")

    print(f"=== DESIGN CASE: EHang Coaxial (Advanced) ===")
    print(f"Target Thrust: {T_target_system:.2f} N (per arm)")
    print(f"Workers: 1 (Serial + Checkpointing)")

    sanity_checks(cfg, base_rot1, base_rot2, fl, r_R, x0, omega_min, omega_max, T_target_system)

    # --- STAGE 1: GLOBAL EVOLUTIONARY SEARCH ---
    if SKIP_STAGE_1 and os.path.exists(checkpoint_file):
        print("\n=== SKIPPING STAGE 1 (Using Checkpoint) ===")
        x1 = x0
    else:
        print("\n=== STAGE 1: Differential Evolution (Global Search) ===")
        t0 = time.time()
        
        # Setup safety callback
        cp_callback = make_checkpoint_callback(checkpoint_file)

        res1 = differential_evolution(
            func=objective_stage1_penalty,
            bounds=bounds,
            args=(base_rot1, base_rot2, fl, cfg.V_design, r_R, cfg, T_target_system),
            strategy='best1bin',
            maxiter=cfg.de_maxiter,
            popsize=cfg.de_popsize,
            tol=0.01,
            mutation=(0.5, 1.0),
            recombination=0.7,
            disp=True,
            workers=1,
            updating='deferred',
            callback=cp_callback 
        )
        t1 = time.time()
        print(f"\nStage 1 Complete in {t1-t0:.1f}s. Message: {res1.message}")
        x1 = res1.x

    # --- STAGE 2: LOCAL GRADIENT REFINEMENT ---
    # IMPORTANT: This block must be OUTSIDE the else!
    print("\n=== STAGE 2: SLSQP Gradient Refinement (Local Polish) ===")
    
    cons_stage2 = [
        {"type": "ineq", "fun": thrust_eq_constraint, "args": (base_rot1, base_rot2, fl, cfg.V_design, r_R, cfg, T_target_system)},
        {"type": "ineq", "fun": tip_mach_ineq_constraint, "args": (base_rot1, base_rot2, r_R, cfg, fl, cfg.M_tip_max)},
    ]

    res2 = minimize(
        fun=objective_power,
        x0=x1,
        args=(base_rot1, base_rot2, fl, cfg.V_design, r_R, cfg),
        method="SLSQP",
        bounds=bounds,
        constraints=cons_stage2,
        options={"maxiter": cfg.stage2_maxiter, "ftol": 1e-6, "disp": True},
    )

    x_opt = res2.x

    # 6. Final Outputs
    totals, out1, out2, coupling, P_shaft, r1_opt, r2_opt = evaluate_system(
        x_opt, base_rot1, base_rot2, fl, cfg.V_design, r_R, cfg
    )

    print("\n=== OPTIMUM SUMMARY ===")
    print(f"P_shaft  : {P_shaft:.2f} W (for one arm)")
    print(f"T_total  : {totals['T_total']:.2f} N (target {T_target_system:.2f} N)")
    print(f"Optimum Ratio (R_upper/R_lower): {x_opt[9]:.4f}")
    print(f"R_upper: {r1_opt.radius:.4f} m")
    print(f"R_lower: {r2_opt.radius:.4f} m")
    
    plot_required_distributions(r_R, r1_opt, r2_opt, out1, out2)
    plot_power_sweep_trimmed(x_opt, base_rot1, base_rot2, fl, r_R, cfg, T_target_system, omega_min, omega_max)
    r1_opt.save_to_yaml(cfg.out_yaml_path)
    print(f"\nSaved to: {cfg.out_yaml_path}")
    
    
if __name__ == "__main__":
    main()