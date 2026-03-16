# -*- coding: utf-8 -*-
"""
=============================================================================
COAXIAL ROTOR OPTIMIZATION - FINAL SUBMISSION
=============================================================================
Description:
    This script optimizes the blade planform (chord and twist) of a coaxial
    rotor system to minimize shaft power during a specific climb flight state.

Theory: 
    It relies on Blade Element Momentum Theory (BEMT) with Prandtl tip-loss 
    factors and a wake interference model to capture the complex aerodynamic 
    interaction between the upper and lower rotors. The optimization is 
    conducted in two stages:
      1. Global Search (Differential Evolution) to find the absolute minimum power.
      2. Local Refinement (COBYLA) to strictly enforce geometric and aerodynamic rules.

Outputs:
    - results/run_YYYYMMDD_HHMMSS/results.npz  (raw data for plotting)
    - results/run_YYYYMMDD_HHMMSS/optimized_propeller.yaml (geometry export)
    - results/run_YYYYMMDD_HHMMSS/summary.txt (human-readable metrics)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
import numpy as np

from scipy.optimize import differential_evolution, minimize, brentq
from scipy.interpolate import PchipInterpolator
import yaml

# --- Custom Physics Modules ---
import fluid
import bemt
import plotting  # Used exclusively at the very end for plotting

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class Config:
    """Stores all constants and design requirements for the optimization."""
    # Aircraft & design state
    mass_kg: float = 650.0
    g: float = 9.81
    altitude_m: float = 500.0
    V_design: float = 3.0           # Climb speed [m/s]

    # Rotor system specifications
    n_coax_units: int = 8           # Number of coaxial pairs
    disk_loading: float = 160.0     # Total system target [N/m^2]
    n_blades: int = 2               # Blades per single rotor
    airfoil_id: int = 2412          # NACA profile

    # Discretization
    n_stations: int = 30            # Radial integration steps
    r_root_norm: float = 0.10       # Root cutout (10% of radius)

    # Coaxial model parameters
    wake_factor: float = 1.3        # Wake contraction/interference severity

    # Optimization settings
    de_maxiter: int = 100           # Stage 1 max iterations
    de_popsize: int = 30            # Stage 1 population diversity
    stage2_maxiter: int = 2500      # Stage 2 strict polishing limit

    # Constraints / aerodynamic limits
    thrust_tol_N: float = 5.0       # Allowed over-thrust margin
    max_c_over_R: float = 0.25      # Absolute max chord width ratio
    alpha_max_deg: float = 12.0     # Stall limit

    # Tip Mach bounds for RPM search
    M_tip_start: float = 0.40
    M_tip_min: float = 0.15
    M_tip_max: float = 0.70

    # Sweep settings (Climb speed variations)
    sweep_delta: int = 10
    sweep_step: int = 1

    # Outputs
    results_root: str = "results"


# =============================================================================
# 2. GEOMETRY KERNELS
# =============================================================================

def radial_grid(n_stations: int, r_root_norm: float) -> np.ndarray:
    """Generates the evaluation points along the blade span."""
    return np.linspace(float(r_root_norm), 1.0, int(n_stations))

def rotor_radius_from_disk_loading(*, T_unit: float, disk_loading: float) -> float:
    """Calculates the physical radius required to meet the disk loading target."""
    return float(np.sqrt(float(T_unit) / (2.0 * np.pi * float(disk_loading))))

def chord_from_ctrl_c_over_R(r_R: np.ndarray, cR_ctrl: np.ndarray) -> np.ndarray:
    """
    Theory: Uses a 4-parameter PCHIP (Cubic) Spline to ensure smooth monotonicity. 
    The second control point is intentionally shifted to 0.35R to force the 
    aerodynamic "Betz hump" where dynamic pressure is highly effective.
    """
    r0 = float(r_R[0])
    r_pts = np.array([r0, 0.35, 0.70, 1.0], dtype=float)
    c_pts = np.array(cR_ctrl, dtype=float)
    cR = PchipInterpolator(r_pts, c_pts)(r_R)
    return np.maximum(cR, 1e-5) # Prevent negative chords

def twist_from_ctrl_deg(r_R: np.ndarray, beta_ctrl_deg: np.ndarray) -> np.ndarray:
    """Evaluates the twist distribution using a 5-point cubic spline."""
    r0 = float(r_R[0])
    r_pts = np.array([r0, 0.30, 0.60, 0.85, 1.0], dtype=float)
    b_pts = np.array(beta_ctrl_deg, dtype=float)
    return PchipInterpolator(r_pts, b_pts)(r_R)


# =============================================================================
# 3. PHYSICS & EVALUATION WRAPPER
# =============================================================================

def evaluate_design(x: np.ndarray, *, cfg: Config, fl: fluid.Fluid, r_R: np.ndarray, T_target_unit: float, V_inf: float) -> dict:
    """
    Decodes the 19 optimization variables into physical dimensions and passes 
    them to the BEMT solver to retrieve aerodynamic performance data.
    """
    x = np.asarray(x, dtype=float)
    
    # 1. Decode Variables
    cU_ctrl = x[0:4]       # Upper Chord control points
    cL_ctrl = x[4:8]       # Lower Chord control points
    betaU_ctrl = x[8:13]   # Upper Twist control points
    betaL_ctrl = x[13:18]  # Lower Twist control points
    omega = float(x[18])   # RPM

    R = rotor_radius_from_disk_loading(T_unit=T_target_unit, disk_loading=cfg.disk_loading)

    # 2. Generate Independent Distributions
    cU_over_R = chord_from_ctrl_c_over_R(r_R, cU_ctrl)
    cL_over_R = chord_from_ctrl_c_over_R(r_R, cL_ctrl)
    chord_U_m = cU_over_R * R
    chord_L_m = cL_over_R * R
    twistU_deg = twist_from_ctrl_deg(r_R, betaU_ctrl)
    twistL_deg = twist_from_ctrl_deg(r_R, betaL_ctrl)

    rho = float(fl.rho)
    a_sound = float(fl.a)
    mu = float(fl.nu * fl.rho)

    # 3. Execute BEMT Physics Engine
    out = bemt.coaxial_bemt_fixed(
        rho=rho, mu=mu, a_sound=a_sound,
        T_total_target=float(T_target_unit),
        R=float(R), B=int(cfg.n_blades), omega=float(omega),
        chord_upper=np.asarray(chord_U_m, dtype=float), 
        chord_lower=np.asarray(chord_L_m, dtype=float),
        twist_upper_deg=np.asarray(twistU_deg, dtype=float),
        twist_lower_deg=np.asarray(twistL_deg, dtype=float),
        V_inf=float(V_inf), airfoil=int(cfg.airfoil_id),
        wake_factor=float(cfg.wake_factor), n_stations=int(cfg.n_stations),
        r_root_cutout=float(cfg.r_root_norm),
    )

    return {
        "R": float(R),
        "omega": float(omega),
        "M_tip": float(omega * R / a_sound),
        "c_over_R_U": np.asarray(cU_over_R, dtype=float), 
        "c_over_R_L": np.asarray(cL_over_R, dtype=float),
        "chord_U_m": np.asarray(chord_U_m, dtype=float),
        "chord_L_m": np.asarray(chord_L_m, dtype=float),
        "twistU_deg": np.asarray(twistU_deg, dtype=float),
        "twistL_deg": np.asarray(twistL_deg, dtype=float),
        "T": float(out["totals"]["T"][0]),
        "P": float(out["totals"]["P"][0]),
        "upper": out["upper"],
        "lower": out["lower"],
    }


# =============================================================================
# 4. OPTIMIZATION STAGE 1 (GLOBAL SEARCH PENALTIES)
# =============================================================================

def enforce_monotone_washout(beta_ctrl_deg: np.ndarray) -> float:
    """Soft penalty preventing the twist from increasing towards the tip."""
    b = np.asarray(beta_ctrl_deg, dtype=float)
    diffs = np.diff(b)  # Should be <= 0
    v = np.maximum(diffs, 0.0)
    return float(np.sum(v * v))

def enforce_chord_taper(cR_ctrl: np.ndarray) -> float:
    """Soft penalty ensuring the chord drops smoothly from peak to tip."""
    c = np.asarray(cR_ctrl, dtype=float)
    diffs = np.diff(c[1:]) # Compare peak vs mid vs tip
    v = np.maximum(diffs, 0.0) 
    return float(np.sum(v * v))

def objective_stage1(x: np.ndarray, *, cfg: Config, fl: fluid.Fluid, r_R: np.ndarray, T_target_unit: float, V_inf: float) -> float:
    """
    Cost function for Differential Evolution. Combines total power with 
    soft penalties to guide the global solver away from impossible designs.
    """
    res = evaluate_design(x, cfg=cfg, fl=fl, r_R=r_R, T_target_unit=T_target_unit, V_inf=V_inf)
    
    P = float(res["P"])
    if (not np.isfinite(P)) or P <= 1.0:
        return 1e12 # Reject immediately
    
    # Soft Thrust Nudge
    penalty_thrust = 100.0 * ((float(res["T"]) - T_target_unit)**2)
    
    # Gentle Stall Barrier
    aU = np.asarray(res["upper"]["alpha_deg"], dtype=float)
    aL = np.asarray(res["lower"]["alpha_deg"], dtype=float)
    alpha_max = float(cfg.alpha_max_deg)
    alpha_min = 0.0 # Prevent negative AoA (windmilling)

    viol_stall_U = np.maximum(0.0, aU - alpha_max)**2 + np.maximum(0.0, alpha_min - aU)**2
    viol_stall_L = np.maximum(0.0, aL - alpha_max)**2 + np.maximum(0.0, alpha_min - aL)**2
    penalty_stall = 1e4 * float(np.mean(viol_stall_U) + np.mean(viol_stall_L))

    # Smoothness / Flat AoA Incentive
    penalty_alpha_var = 1e4 * (float(np.var(aU)) + float(np.var(aL)))

    # Geometric Sanity
    cU_ctrl, cL_ctrl = x[0:4], x[4:8]
    betaU_ctrl, betaL_ctrl = x[8:13], x[13:18]
    penalty_geom = 1e4 * (enforce_monotone_washout(betaU_ctrl) + enforce_monotone_washout(betaL_ctrl) + 
                          enforce_chord_taper(cU_ctrl) + enforce_chord_taper(cL_ctrl))
    
    return float(P + penalty_thrust + penalty_stall + penalty_alpha_var + penalty_geom)


# =============================================================================
# 5. DATA EXTRACTION & EXPORT HELPERS
# =============================================================================

def calculate_blade_loading(res: dict, cfg: Config, fl: fluid.Fluid, r_R: np.ndarray) -> dict:
    """
    Theory: Calculates Solidity (sigma) and Blade Loading (C_T / sigma) for 
    the upper, lower, and total rotor system. Target design is 0.1 to balance 
    stall margin against profile drag.
    """
    rho = float(fl.rho)
    R_val = float(res["R"])
    A_disk = np.pi * (R_val**2)
    V_tip = res["omega"] * R_val
    
    # Thrust Coefficients (C_T)
    C_T_total = res["T"] / (rho * A_disk * (V_tip**2))
    C_T_upper = res["upper"]["T"][0] / (rho * A_disk * (V_tip**2))
    C_T_lower = res["lower"]["T"][0] / (rho * A_disk * (V_tip**2))
    
    # Solidity (sigma) using trapezoidal integration for precise area
    c_avg_upper = np.trapezoid(res["chord_U_m"], r_R * R_val) / R_val
    c_avg_lower = np.trapezoid(res["chord_L_m"], r_R * R_val) / R_val
    sigma_upper = (cfg.n_blades * c_avg_upper) / (np.pi * R_val)
    sigma_lower = (cfg.n_blades * c_avg_lower) / (np.pi * R_val)
    sigma_total = sigma_upper + sigma_lower
    
    return {
        "sigma_upper": sigma_upper, "sigma_lower": sigma_lower, "sigma_total": sigma_total,
        "C_T_upper": C_T_upper, "C_T_lower": C_T_lower, "C_T_total": C_T_total,
        "bl_upper": C_T_upper / sigma_upper,
        "bl_lower": C_T_lower / sigma_lower,
        "bl_total": C_T_total / sigma_total
    }

def save_results_to_yaml(filepath: str, cfg: Config, final: dict, r_R: np.ndarray):
    """Exports geometry and flight state for use in other software."""
    data = {
        "design_state": {
            "mass_kg": float(cfg.mass_kg),
            "climb_speed_m_s": float(cfg.V_design),
            "thrust_target_N": float(final["T"]),
            "shaft_power_W": float(final["P"])
        },
        "rotor_geometry": {
            "radius_m": float(final["R"]),
            "omega_rad_s": float(final["omega"]),
        },
        "blade_distributions": {
            "stations_r_R": [float(x) for x in r_R],
            "chord_upper_m": [float(x) for x in final["chord_U_m"]],
            "chord_lower_m": [float(x) for x in final["chord_L_m"]],
            "twist_upper_deg": [float(x) for x in final["twistU_deg"]],
            "twist_lower_deg": [float(x) for x in final["twistL_deg"]]
        }
    }
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# =============================================================================
# 6. MAIN OPTIMIZATION LOOP
# =============================================================================

def trim_omega_for_thrust(x_geom, *, cfg, fl, r_R, T_target_unit, V_inf, omega_lo, omega_hi, omega_init):
    """Helper to find the required RPM for a given flight speed via Root Finding."""
    def thrust_err(omega: float):
        xx = x_geom.copy(); xx[18] = float(omega)
        res = evaluate_design(xx, cfg=cfg, fl=fl, r_R=r_R, T_target_unit=T_target_unit, V_inf=V_inf)
        return float(res["T"] - T_target_unit), res

    lo, hi = max(omega_lo, omega_init / 1.6), min(omega_hi, omega_init * 1.6)
    f_lo, _ = thrust_err(lo)
    f_hi, _ = thrust_err(hi)

    expand = 0
    while f_lo * f_hi > 0.0 and expand < 10:
        lo, hi = max(omega_lo, lo / 1.6), min(omega_hi, hi * 1.6)
        f_lo, _ = thrust_err(lo)
        f_hi, _ = thrust_err(hi)
        expand += 1

    if f_lo * f_hi > 0.0:
        return (lo, thrust_err(lo)[1]) if abs(f_lo) < abs(f_hi) else (hi, thrust_err(hi)[1])

    omega_trim = float(brentq(lambda om: float(thrust_err(om)[0]), lo, hi, xtol=1e-6))
    return omega_trim, thrust_err(omega_trim)[1]


def main():
    cfg = Config()
    fl = fluid.Fluid(cfg.altitude_m)
    W = float(cfg.mass_kg * cfg.g)
    T_target_unit = float(W / cfg.n_coax_units)

    r_R = radial_grid(cfg.n_stations, cfg.r_root_norm)
    R = rotor_radius_from_disk_loading(T_unit=T_target_unit, disk_loading=cfg.disk_loading)

    omega_lo = float(cfg.M_tip_min * fl.a / R)
    omega_hi = float(cfg.M_tip_max * fl.a / R)
    omega_start = float(cfg.M_tip_start * fl.a / R)

    # --- DEFINE BOUNDS ---
    
    lb = (
        # Upper chord [m] (Root, Peak, Mid, Tip)
        [0.050, 0.100, 0.075, 0.020] +
        # Lower chord [m]
        [0.050, 0.100, 0.075, 0.020] +
        # Upper twist [deg] (root -> tip control points)
        [18.0, 16.0, 13.0, 10.0,  6.0] +
        # Lower twist [deg]
        [16.0, 14.0, 12.0,  9.0,  5.0] +
        # Omega [rad/s]
        [omega_lo]
)

    ub = (
        # Upper chord [m]
        [0.095, 0.180, 0.140, 0.070] +
        # Lower chord [m]
        [0.095, 0.180, 0.140, 0.070] +
        # Upper twist [deg]
        [35.0, 30.0, 24.0, 18.0, 14.0] +
        # Lower twist [deg]
        [32.0, 28.0, 22.0, 17.0, 13.0] +
        # Omega [rad/s]
        [omega_hi]
        )


    # Start Seed: High-quality textbook shape ensures fast convergence
    x0 = np.array([
        0.08, 0.12, 0.076, 0.04,        # Upper Chord
        0.08, 0.11, 0.076, 0.04,        # Lower Chord 
        22.0, 18.0, 14.0, 10.5, 8.0,    # Upper Twist
        24.0, 20.0, 15.0, 11.0, 9.0,    # Lower Twist 
        omega_start                     # RPM
    ], dtype=float)



    # --- STAGE 1: GLOBAL OPTIMIZATION (DE) ---
    print("=== STAGE 1: Differential Evolution ===")
    obj1 = lambda x: objective_stage1(x, cfg=cfg, fl=fl, r_R=r_R, T_target_unit=T_target_unit, V_inf=cfg.V_design)
    res1 = differential_evolution(
        func=obj1, bounds=list(zip(lb, ub)), strategy="best1bin",
        maxiter=cfg.de_maxiter, popsize=cfg.de_popsize, x0=x0, tol=0.01, disp=True, polish=False, updating= "deferred", workers=1
    )
    x1 = np.asarray(res1.x, dtype=float)
    
    # --- STAGE 2: LOCAL OPTIMIZATION & CONSTRAINTS (COBYLA) ---
    print("\n=== STAGE 2: COBYLA Polish ===")
    
    # Helper to clean up constraint functions
    def eval_x(x): return evaluate_design(x, cfg=cfg, fl=fl, r_R=r_R, T_target_unit=T_target_unit, V_inf=cfg.V_design)
    
    cons = [
        # Thrust Band
        {"type":"ineq","fun": lambda x: eval_x(x)["T"] - T_target_unit},
        {"type":"ineq","fun": lambda x: (T_target_unit + cfg.thrust_tol_N) - eval_x(x)["T"]},
        # Strict Structural Taper (Rise then Taper rule)
        {"type":"ineq","fun": lambda x: float(min(x[1]-x[0], x[1]-x[2], x[2]-x[3]))}, # Upper
        {"type":"ineq","fun": lambda x: float(min(x[5]-x[4], x[5]-x[6], x[6]-x[7]))}, # Lower
        # Torque Balance (20%)
        {"type":"ineq","fun": lambda x: 0.20 - (abs(abs(eval_x(x)["upper"]["Q"][0]) - abs(eval_x(x)["lower"]["Q"][0])) / max(abs(eval_x(x)["upper"]["Q"][0]) + abs(eval_x(x)["lower"]["Q"][0]), 1e-6))},
        # Blade Loading Band (0.098 to 0.105)
        {"type":"ineq","fun": lambda x: calculate_blade_loading(eval_x(x), cfg, fl, r_R)["bl_total"] - 0.098},
        {"type":"ineq","fun": lambda x: 0.105 - calculate_blade_loading(eval_x(x), cfg, fl, r_R)["bl_total"]}
    ]
    
    # --- MASKED STALL & WINDMILLING LIMITS (Only enforce on working span, r/R > 0.2) ---
    def c_alpha_upper_max(x):
        aU = np.asarray(eval_x(x)["upper"]["alpha_deg"], dtype=float)
        return float(cfg.alpha_max_deg - np.max(np.abs(aU[r_R > 0.2])))

    def c_alpha_lower_max(x):
        aL = np.asarray(eval_x(x)["lower"]["alpha_deg"], dtype=float)
        return float(cfg.alpha_max_deg - np.max(np.abs(aL[r_R > 0.2])))
        
    def c_alpha_upper_min(x):
        aU = np.asarray(eval_x(x)["upper"]["alpha_deg"], dtype=float)
        return float(np.min(aU[r_R > 0.2]) - 0.0) # Must be >= 0.0

    def c_alpha_lower_min(x):
        aL = np.asarray(eval_x(x)["lower"]["alpha_deg"], dtype=float)
        return float(np.min(aL[r_R > 0.2]) - 0.0) # Must be >= 0.0

    cons += [
        {"type":"ineq","fun": c_alpha_upper_max}, 
        {"type":"ineq","fun": c_alpha_lower_max},
        {"type":"ineq","fun": c_alpha_upper_min}, 
        {"type":"ineq","fun": c_alpha_lower_min}
    ]
    # ---------------------------------------------------------------------------------

    # Enforce standard bounds inside COBYLA
    for i in range(len(x1)):
        cons.append({"type":"ineq","fun": lambda x, i=i: float(x[i] - lb[i])})
        cons.append({"type":"ineq","fun": lambda x, i=i: float(ub[i] - x[i])})

    def obj2(x):
        res = eval_x(x); P = float(res["P"])
        print(f"COBYLA step: P={P:.1f} W | T={res['T']:.1f} N | Mtip={res['M_tip']:.3f}", flush=True)
        return P if np.isfinite(P) and P > 1.0 else 1e12

    res2 = minimize(fun=obj2, x0=x1, method="COBYLA", constraints=cons, options={"maxiter": cfg.stage2_maxiter, "rhobeg": 0.02, "disp": True})
    x_opt = np.asarray(res2.x, dtype=float)
    final = eval_x(x_opt)

    # --- STAGE 3: CLIMB SPEED SWEEP ---
    # Hardcoded range: -10 m/s (descent) to +10 m/s (climb) in 1 m/s steps
    Vs = np.arange(-10.0, 11.0, 1.0, dtype=float)
    P_sw, P_ex_sw, RPM_sw, M_sw, T_sw = [], [], [], [], []
    for V in Vs:
        om_trim, res_trim = trim_omega_for_thrust(x_geom=x_opt.copy(), cfg=cfg, fl=fl, r_R=r_R, 
                                                  T_target_unit=T_target_unit, V_inf=float(V), 
                                                  omega_lo=omega_lo, omega_hi=omega_hi, omega_init=float(final["omega"]))
        P_sw.append(res_trim["P"])
        P_ex_sw.append(res_trim["P"] - res_trim["T"] * float(V)) # Aerodynamic power
        RPM_sw.append(om_trim * 30.0 / np.pi)
        M_sw.append(res_trim["M_tip"])
        T_sw.append(res_trim["T"])

    # --- DATA EXPORT ---
    run_dir = os.path.join(cfg.results_root, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    bl_data = calculate_blade_loading(final, cfg, fl, r_R)



    # --- ADDITIONAL CALCULATIONS FOR REPORT ---
    # 1. Thrust Breakdown
    T_upper = float(final["upper"]["T"][0])
    T_lower = float(final["lower"]["T"][0])
    T_total_unit = final["T"]
    T_total_aircraft = T_total_unit * cfg.n_coax_units
    thrust_error_pct = abs(T_total_unit - T_target_unit) / T_target_unit * 100.0

    # 2. Power Breakdown
    P_upper = float(final["upper"]["P"][0])
    P_lower = float(final["lower"]["P"][0])
    P_total_unit = final["P"]
    P_total_aircraft = P_total_unit * cfg.n_coax_units

    # 3. IDEAL COMPARISON & EFFICIENCY (Apples-to-Apples Coaxial Baseline)
    A_disk = np.pi * (final["R"]**2)
    rho = float(fl.rho)
    T_upper = float(final["upper"]["T"][0])
    T_lower = float(final["lower"]["T"][0])
    T_total = float(final["T"])
    V = float(cfg.V_design)
    P_total_unit = float(final["P"])

    # Useful (climb) power
    P_useful = T_total * V

    # Aerodynamic power actually lost to induced+profile
    P_aero = P_total_unit - P_useful

    # Single disk ideal (for reference only, assumes one disk takes all thrust)
    P_ideal_single = (T_total**1.5) / np.sqrt(2.0 * rho * A_disk)
    
    # Isolated Pair Ideal (The correct lower bound for a coaxial system)
    # Assumes two rotors sharing the thrust, but infinitely far apart (no wake overlap)
    P_ideal_pair = (T_upper**1.5 + T_lower**1.5) / np.sqrt(2.0 * rho * A_disk)

    # Hover-style figure of merit (benchmark vs aerodynamic power using pair ideal)
    FM_pair_style = P_ideal_pair / max(P_aero, 1e-9)

    # Propulsive efficiency in climb
    eta_p = P_useful / max(P_total_unit, 1e-9)
    
    # Power loading (Converted to N/kW for standard industry readability)
    power_loading_N_kW = 1000.0 * T_total_unit / P_total_unit

    # 4. Operating Point 
    RPM = final["omega"] * 30.0 / np.pi
    # Max Mach usually happens at the tip
    W_tip_upper = np.sqrt((final["omega"] * final["R"])**2 + final["upper"]["Vax"][-1]**2)
    M_max = W_tip_upper / fl.a
    # Max Reynolds usually happens at the thickest part of the hump
    chord_max = np.max(final["chord_U_m"])
    W_hump = np.sqrt((final["omega"] * 0.35 * final["R"])**2 + final["upper"]["Vax"][int(0.35*cfg.n_stations)]**2)
    Re_max = rho * W_hump * chord_max / (fl.nu * rho)

    # 5. Aerodynamic Health
    aU = final["upper"]["alpha_deg"]
    aL = final["lower"]["alpha_deg"]
    cl_max = max(np.max(final["upper"]["CL"]), np.max(final["lower"]["CL"]))
    stall_status = "No stall detected" if cl_max < 1.2 else "WARNING: Potential Stall"

    # --- BUILD THE SUMMARY STRING ---
    summary_lines = [
        "DESIGN CASE",
        "------------",
        f"Altitude: {cfg.altitude_m:.0f} m",
        f"Density: {rho:.3f} kg/m³",
        f"Climb speed: {cfg.V_design:.0f} m/s",
        f"Aircraft mass: {cfg.mass_kg:.0f} kg",
        f"Rotors: {cfg.n_coax_units} coaxial ({cfg.n_coax_units*2} disks)",
        f"Disk loading: {cfg.disk_loading:.0f} N/m²",
        f"Root cutout: {cfg.r_root_norm}R",
        "",
        "THRUST",
        "------",
        f"Target thrust per coaxial unit (pair): {T_target_unit:.0f} N",
        f"Target thrust per disk: {0.5 * T_target_unit:.0f} N",
        f"Upper thrust: {T_upper:.0f} N",
        f"Lower thrust: {T_lower:.0f} N",
        f"Total per coaxial unit: {T_total_unit:.0f} N",
        f"Total aircraft thrust: {T_total_aircraft:.0f} N",
        f"Thrust error: {thrust_error_pct:.2f}%",
        "",
        "POWER",
        "------",
        f"Upper power: {P_upper:.0f} W",
        f"Lower power: {P_lower:.0f} W",
        f"Total per coaxial: {P_total_unit:.0f} W",
        f"Total aircraft: {P_total_aircraft:.0f} W",
        "",
        "IDEAL COMPARISON & EFFICIENCY",
        "-----------------------------",
        f"Useful climb power:   {P_useful:.0f} W",
        f"Aerodynamic power:    {P_aero:.0f} W",
        f"Ideal hover (Single): {P_ideal_single:.0f} W",
        f"Ideal hover (Pair):   {P_ideal_pair:.0f} W",
        f"Hover-style FM:       {FM_pair_style:.3f}",
        f"Propulsive Eff (ηp):  {eta_p * 100:.1f}%",
        f"Power loading:        {power_loading_N_kW:.1f} N/kW",
        "",
        "OPERATING POINT",
        "---------------",
        f"Omega: {final['omega']:.0f} rad/s",
        f"RPM: {RPM:.0f}",
        f"Tip Mach: {final['M_tip']:.2f}",
        f"Max Mach: {M_max:.2f}",
        f"Max Reynolds: {Re_max:.1e}",
        "",
        "AERODYNAMIC HEALTH",
        "------------------",
        f"AoA upper: {np.min(aU):.1f}° – {np.max(aU):.1f}°",
        f"AoA lower: {np.min(aL):.1f}° – {np.max(aL):.1f}°",
        f"Max CL: {cl_max:.2f}",
        f"{stall_status}",
        "",
        "BLADE LOADING CHECK",
        "-------------------",
        f"Solidity (Upper/Lower): {bl_data['sigma_upper']:.4f} / {bl_data['sigma_lower']:.4f}",
        f"System Blade Loading:   {bl_data['bl_total']:.4f}",
        f"Stage2 Success:         {bool(res2.success)}"
    ]
    
    with open(os.path.join(run_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    npz_path = os.path.join(run_dir, "results.npz")
    upper = final["upper"]
    lower = final["lower"]

    np.savez(
        npz_path,
        # 1. Geometry & State (For Planform, Chord, and Twist plots)
        R=final["R"],
        omega=final["omega"],
        r_R=r_R,
        V_design=cfg.V_design,
        chord_U_m=final["chord_U_m"],
        chord_L_m=final["chord_L_m"],
        c_over_R_U=final["c_over_R_U"],
        c_over_R_L=final["c_over_R_L"],
        twistU_deg=final["twistU_deg"],
        twistL_deg=final["twistL_deg"],

        # 2. Aerodynamic Angles (For Inflow and AoA plots)
        phiU_deg=np.degrees(upper["phi"]),
        phiL_deg=np.degrees(lower["phi"]),
        alphaU_deg=upper["alpha_deg"],
        alphaL_deg=lower["alpha_deg"],

        # 3. Forces & Loading (For Thrust and Torque loading plots)
        dTdrU=upper["dTdr"],
        dTdrL=lower["dTdr"],
        dQdrU=upper["dQdr"],
        dQdrL=lower["dQdr"],

        # 4. Coefficients (For CL and CD plots)
        CL_upper=upper["CL"],
        CD_upper=upper["CD"],
        CL_lower=lower["CL"],
        CD_lower=lower["CD"],

        # 5. Velocities (For Axial Velocity plots)
        VaxU=upper["Vax"],
        VaxL=lower["Vax"],

        # 6. Performance Totals
        T_design=final["T"],
        P_design=final["P"],

        # 7. Sweep Data (For Power vs. Climb Speed plots)
        Vs=Vs,
        P_sweep=np.asarray(P_sw),
        P_excess_sweep=np.asarray(P_ex_sw),
        RPM_sweep=np.asarray(RPM_sw),
        T_sweep=np.asarray(T_sw),
        Mtip_sweep=np.asarray(M_sw)
    )

    save_results_to_yaml(os.path.join(run_dir, "optimized_propeller.yaml"), cfg, final, r_R)
    plotting.make_required_plots(npz_path, run_dir)

    print("\n=== OUTPUTS ===")
    print(f"Run folder : {run_dir}\nSummary    : {os.path.join(run_dir, 'summary.txt')}\nPNGs       : Saved successfully.")

if __name__ == "__main__":
    main()