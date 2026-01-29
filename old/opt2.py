import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds
from scipy.interpolate import CubicSpline

import fluid
import propeller
import bemt_coaxial2


# =========================
# FINAL REPORT INPUTS
# =========================
MASS_KG = 2500.0
G = 9.81

ALTITUDE_M = 500.0
V_CLIMB = 3.0

DISK_LOADING = 200.0          # N/m^2
N_COAX_ROTORS = 2             # coaxial pair (upper+lower)
N_PROPULSORS = 1              # number of coaxial systems on aircraft

M_TIP_MAX = 0.63
N_BLADES_REQUIRED = 4

N_STATIONS = 30
R_ROOT_NORM = 0.10            # REQUIRED: root cutout = 0.1R

# We are NOT allowed to modify the original YAML on disk.
# But to make dimensions physically meaningful for a ~4.4 m rotor, we set c_ref in memory.
C_REF_OVERRIDE_M = 0.30

# If you must force NACA2412 explicitly:
# If your code uses airfoil numeric IDs, set FORCE_AIRFOIL to that integer (e.g. 2412)
# If you do NOT know the ID mapping, keep None and it will use YAML's first airfoil along span.
FORCE_AIRFOIL = 2412   # set to None to keep YAML first ID, or set to 2412 if your model expects NACA code


# =========================
# Geometry parameterization
# =========================
def radial_grid(n_stations, r_root_norm):
    return np.linspace(float(r_root_norm), 1.0, int(n_stations))

def chord_cubic_spline(r_R, c_ctrl):
    """
    4-parameter cubic spline chord distribution.
    c_ctrl are chord/R or chord-norm values (consistent with your prop model usage).
    """
    r_pts = np.array([r_R[0], 0.45, 0.75, 1.0], dtype=float)
    c_pts = np.array(c_ctrl, dtype=float)
    cs = CubicSpline(r_pts, c_pts, bc_type="natural")
    c = cs(r_R)
    return np.maximum(c, 1e-4)

def linear_pitch(r_R, beta_root, beta_tip):
    r0 = float(r_R[0])
    r1 = float(r_R[-1])
    t = (r_R - r0) / max(r1 - r0, 1e-12)
    return beta_root + (beta_tip - beta_root) * t

def compute_required_radius_from_disk_loading(mass_kg, disk_loading, n_coax_rotors=2):
    """
    Disk loading uses total area of coaxial rotors:
      DL = W / (n_coax_rotors * pi * R^2)
    => R = sqrt( W / (n_coax_rotors*pi*DL) )
    """
    W = float(mass_kg) * float(G)
    return np.sqrt(W / (float(n_coax_rotors) * np.pi * float(disk_loading)))


def override_scale_in_memory(prop, R_new, c_ref_new, nblades_new):
    """
    Override radius, c_ref, nblades in memory (do NOT touch YAML on disk).
    """
    prop.radius = float(R_new)
    prop.c_ref = float(c_ref_new)
    prop.nblades = int(nblades_new)
    prop.omega = float(prop.omega)  # keep omega (will be overridden by optimization anyway)
    return prop


def build_rotors_from_x(x, base_rot1, base_rot2, r_R):
    """
    x =
      [0..3] chord ctrl (4 params)
      [4] beta1_root
      [5] beta1_tip
      [6] beta2_root
      [7] beta2_tip
      [8] omega (rad/s) (same omega for both)
    """
    c_ctrl = x[0:4]
    b1_root, b1_tip = x[4], x[5]
    b2_root, b2_tip = x[6], x[7]
    omega = float(x[8])

    # chord distribution (your propeller model treats prop.chord as normalized; bemt_single multiplies by c_ref)
    chord_dist = chord_cubic_spline(r_R, c_ctrl)

    # pitch distribution in degrees
    beta1 = linear_pitch(r_R, b1_root, b1_tip)
    beta2 = linear_pitch(r_R, b2_root, b2_tip)

    r1 = copy.deepcopy(base_rot1)
    r2 = copy.deepcopy(base_rot2)

    r1.r_R = np.array(r_R, dtype=float)
    r2.r_R = np.array(r_R, dtype=float)

    r1.chord = np.array(chord_dist, dtype=float)
    r2.chord = np.array(chord_dist, dtype=float)

    r1.pitch = np.array(beta1, dtype=float)
    r2.pitch = np.array(beta2, dtype=float)

    # Force airfoil constant along span (optional)
    if FORCE_AIRFOIL is not None:
        r1.airfoil = np.full_like(r1.r_R, int(FORCE_AIRFOIL), dtype=int)
        r2.airfoil = np.full_like(r2.r_R, int(FORCE_AIRFOIL), dtype=int)
    else:
        # if YAML has airfoil list, keep first value constant along span
        if hasattr(r1, "airfoil") and len(r1.airfoil) > 0:
            r1.airfoil = np.full_like(r1.r_R, int(r1.airfoil[0]), dtype=int)
            r2.airfoil = np.full_like(r2.r_R, int(r2.airfoil[0]), dtype=int)

    r1.omega = omega
    r2.omega = omega

    return r1, r2


# =========================
# Model evaluation
# =========================
def evaluate_system(x, base_rot1, base_rot2, fl, V_inf, r_R):
    r1, r2 = build_rotors_from_x(x, base_rot1, base_rot2, r_R)

    totals, out1, out2, coupling = bemt_coaxial2.coaxial_bemt_fixed(
        r1, r2, fl,
        V_inf=V_inf,
        omega1=r1.omega,
        omega2=r2.omega,
        trim_rotor2=False,      # do NOT trim during optimization
        alpha_target_deg=3.0
    )

    omega = float(x[8])
    P_shaft = float(totals["Q_total"] * omega)
    return totals, out1, out2, coupling, P_shaft, r1, r2


# =========================
# Optimization objective/constraints
# =========================
def objective_power(x, base_rot1, base_rot2, fl, V_inf, r_R):
    """
    Stage-2 objective: minimize shaft power (Q*Omega) at thrust equality constraint.
    """
    try:
        totals, _, _, _, P, _, _ = evaluate_system(x, base_rot1, base_rot2, fl, V_inf, r_R)
        if (not np.isfinite(P)) or P <= 0:
            return 1e12
        # discourage negative thrust (windmilling)
        if totals["T1"] <= 0 or totals["T2"] <= 0:
            return float(P + 1e8)
        return float(P)
    except Exception:
        return 1e12

def objective_stage1_penalty(x, base_rot1, base_rot2, fl, V_inf, r_R, T_target):
    """
    Stage-1 objective: power + strong penalty on thrust mismatch to get near feasibility.
    """
    try:
        totals, _, _, _, P, _, _ = evaluate_system(x, base_rot1, base_rot2, fl, V_inf, r_R)
        if (not np.isfinite(P)) or P <= 0:
            return 1e12
        T = float(totals["T_total"])
        err = (T - T_target) / max(T_target, 1e-12)
        return float(P + 1e9 * err * err)
    except Exception:
        return 1e12

def thrust_eq_constraint(x, base_rot1, base_rot2, fl, V_inf, r_R, T_target):
    try:
        totals, _, _, _, _, _, _ = evaluate_system(x, base_rot1, base_rot2, fl, V_inf, r_R)
        return float(totals["T_total"] - T_target)
    except Exception:
        return 1e9

def tip_mach_ineq_constraint(x, fl, R_tip, M_tip_max):
    """
    inequality constraint: M_tip_max - M_tip >= 0
    """
    try:
        omega = float(x[8])
        M_tip = omega * float(R_tip) / float(fl.a)
        return float(M_tip_max - M_tip)
    except Exception:
        return -1e9


# =========================
# Plotting
# =========================
def plot_geometry_and_loads(r_R, r1, r2, out1, out2):
    fig, axs = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    axs[0].plot(r_R, r1.pitch, label="Rotor 1 pitch [deg]")
    axs[0].plot(r_R, r2.pitch, label="Rotor 2 pitch [deg]")
    axs[0].set_ylabel("Pitch [deg]")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(r_R, np.asarray(r1.chord) * float(r1.c_ref), label="Chord [m]")
    axs[1].set_ylabel("Chord [m]")
    axs[1].grid(True)
    axs[1].legend()

    axs[2].plot(r_R, out1["dT_dr"], label="Rotor 1 dT/dr")
    axs[2].plot(r_R, out2["dT_dr"], label="Rotor 2 dT/dr")
    axs[2].set_xlabel("r/R [-]")
    axs[2].set_ylabel("dT/dr [N/m]")
    axs[2].grid(True)
    axs[2].legend()

    plt.tight_layout()
    plt.show()


# =========================
# REQUIRED: RPM-trimmed sweep (GUARANTEED TERMINATION)
# =========================
def trim_omega_for_thrust_capped(
    x_base, base_rot1, base_rot2, fl, V_inf, r_R,
    T_target,
    omega_min, omega_max,
    max_bisect_iter=25,
    n_scan=21,
    thrust_tol=1e-2
):
    """
    Guaranteed-termination omega trim:
      1) Try bisection if bracketed (hard cap max_bisect_iter)
      2) If no bracket or non-finite, do a fixed scan (n_scan)
      3) If still fails, return omega0 (x_base[8])

    Returns: omega_trim, totals, P_shaft
    """
    x_tmp = np.array(x_base, dtype=float).copy()
    omega0 = float(x_tmp[8])

    # Keep bounds reasonable around omega0 to avoid pathological BEMT points
    om_min = float(max(omega_min, 0.60 * omega0))
    om_max = float(min(omega_max, 1.40 * omega0))
    if om_max <= om_min:
        om_min = float(max(omega_min, 0.80 * omega0))
        om_max = float(min(omega_max, 1.20 * omega0))

    def eval_f(omega):
        try:
            x_tmp[8] = float(omega)
            totals, _, _, _, P, _, _ = evaluate_system(x_tmp, base_rot1, base_rot2, fl, V_inf, r_R)
            f = float(totals["T_total"]) - float(T_target)
            if (not np.isfinite(f)) or (not np.isfinite(P)):
                return np.nan, totals, np.nan
            return f, totals, float(P)
        except Exception:
            return np.nan, {"T_total": np.nan}, np.nan

    f_lo, tot_lo, P_lo = eval_f(om_min)
    f_hi, tot_hi, P_hi = eval_f(om_max)

    if np.isfinite(f_lo) and abs(f_lo) < thrust_tol:
        return om_min, tot_lo, P_lo
    if np.isfinite(f_hi) and abs(f_hi) < thrust_tol:
        return om_max, tot_hi, P_hi

    # 1) Bisection if bracketed
    if np.isfinite(f_lo) and np.isfinite(f_hi) and (f_lo * f_hi < 0.0):
        lo, hi = om_min, om_max
        flo, fhi = f_lo, f_hi
        best = None  # (abs_f, omega, totals, P)

        for _ in range(int(max_bisect_iter)):
            mid = 0.5 * (lo + hi)
            fmid, tot_mid, P_mid = eval_f(mid)
            if np.isfinite(fmid):
                af = abs(float(fmid))
                if (best is None) or (af < best[0]):
                    best = (af, mid, tot_mid, P_mid)
                if af < thrust_tol:
                    return mid, tot_mid, P_mid
                if flo * fmid < 0.0:
                    hi, fhi = mid, fmid
                else:
                    lo, flo = mid, fmid
            else:
                # shrink interval deterministically; guarantees termination
                hi = mid

        if best is not None:
            return best[1], best[2], best[3]

    # 2) Fallback scan
    omegas = np.linspace(om_min, om_max, int(n_scan))
    best = None  # (abs_f, omega, totals, P)
    for om in omegas:
        fval, tot, P = eval_f(om)
        if not np.isfinite(fval):
            continue
        af = abs(float(fval))
        if (best is None) or (af < best[0]):
            best = (af, float(om), tot, float(P))
            if af < thrust_tol:
                break

    if best is not None:
        return best[1], best[2], best[3]

    # 3) last-resort
    f0, tot0, P0 = eval_f(omega0)
    return omega0, tot0, P0


def plot_power_sweep_trimmed(x_opt, base_rot1, base_rot2, fl, r_R, V_design, T_target, omega_min, omega_max):
    speeds = np.arange(V_design - 10.0, V_design + 10.1, 1.0)

    P_shaft_list = []
    P_aero_list = []
    T_list = []
    omega_list = []

    for V in speeds:
        print(f"[SWEEP] V={V:+.1f} m/s -> trimming omega...")

        omega_trim, totals_s, P_s = trim_omega_for_thrust_capped(
            x_opt, base_rot1, base_rot2, fl, V, r_R,
            T_target=T_target,
            omega_min=omega_min,
            omega_max=omega_max,
            max_bisect_iter=25,
            n_scan=21,
            thrust_tol=1e-2
        )

        T_s = float(totals_s.get("T_total", np.nan))
        P_shaft_list.append(P_s)
        P_aero_list.append(P_s - T_s * V if (np.isfinite(P_s) and np.isfinite(T_s)) else np.nan)
        T_list.append(T_s)
        omega_list.append(float(omega_trim))

    plt.figure(figsize=(9, 5))
    plt.plot(speeds, P_shaft_list, marker="o", label="P_shaft = QΩ (including climb power)")
    plt.plot(speeds, P_aero_list, marker="s", label="P_aero = QΩ - T·V (excluding climb power)")
    plt.axvline(V_design, linestyle="--")
    plt.xlabel("Climb Speed V_inf [m/s]")
    plt.ylabel("Power [W]")
    plt.title("Power vs Climb Speed (RPM-trimmed to match thrust)")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(9, 4))
    plt.plot(speeds, omega_list, marker="o")
    plt.axvline(V_design, linestyle="--")
    plt.xlabel("Climb Speed V_inf [m/s]")
    plt.ylabel("Trimmed ω [rad/s]")
    plt.title("Trimmed ω vs Climb Speed")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(9, 4))
    plt.plot(speeds, T_list, marker="o", label="Achieved T_total")
    plt.axhline(T_target, linestyle="--", label="Target thrust")
    plt.axvline(V_design, linestyle="--")
    plt.xlabel("Climb Speed V_inf [m/s]")
    plt.ylabel("Thrust [N]")
    plt.title("Thrust Tracking Check (after RPM trim)")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_blade_planform(rot, title="Blade planform (optimized geometry)"):

    r_R = np.asarray(rot.r_R, dtype=float)
    r = r_R * float(rot.radius)

    c = np.asarray(rot.chord, dtype=float) * float(rot.c_ref)
    x_le = -0.75 * c
    x_te = +0.25 * c

    r_f = np.linspace(r.min(), r.max(), 400)
    le = CubicSpline(r, x_le, bc_type="natural")(r_f)
    te = CubicSpline(r, x_te, bc_type="natural")(r_f)

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.plot(r_f, le, label="LE — spline")
    ax.plot(r_f, te, label="TE — spline")
    ax.fill_between(r_f, le, te, alpha=0.2)
    ax.axhline(0.0, linestyle="--", linewidth=0.8)
    ax.set_xlabel("r [m]")
    ax.set_ylabel("x [m]")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    plt.show()


# =========================
# Main
# =========================
def main():
    yaml_path = os.path.join("data", "pybemt_tmotor28.yaml")
    base_prop = propeller.Propeller.load_from_yaml(yaml_path)

    # Compute rotor radius from disk loading (coaxial total area)
    R_required = compute_required_radius_from_disk_loading(MASS_KG, DISK_LOADING, n_coax_rotors=N_COAX_ROTORS)

    W = MASS_KG * G
    T_target_system = W / float(N_PROPULSORS)

    fl = fluid.Fluid(ALTITUDE_M)

    print("=== FINAL REPORT DESIGN CASE ===")
    print(f"Mass: {MASS_KG:.1f} kg -> Weight: {W:.1f} N")
    print(f"Altitude: {ALTITUDE_M:.1f} m")
    print(f"Climb speed: {V_CLIMB:.2f} m/s")
    print(f"Disk loading: {DISK_LOADING:.1f} N/m^2 (using total area of {N_COAX_ROTORS} rotors)")
    print(f"Computed rotor radius: R = {R_required:.3f} m  (R_upper/R_lower=1)")
    print(f"Target thrust per coaxial system: {T_target_system:.1f} N (N_propulsors={N_PROPULSORS})")
    print(f"Tip Mach limit: {M_TIP_MAX:.3f}")
    print(f"Stations: {N_STATIONS}")
    print(f"Root cutout: {R_ROOT_NORM:.2f}R")

    if int(getattr(base_prop, "nblades", N_BLADES_REQUIRED)) != N_BLADES_REQUIRED:
        print(f"[Warning] YAML has nblades={base_prop.nblades}, but requirement is {N_BLADES_REQUIRED}. Forcing in code.")

    # Build two rotors (in memory only) and override scale + blade count
    base_rot1 = override_scale_in_memory(copy.deepcopy(base_prop), R_required, C_REF_OVERRIDE_M, N_BLADES_REQUIRED)
    base_rot2 = override_scale_in_memory(copy.deepcopy(base_prop), R_required, C_REF_OVERRIDE_M, N_BLADES_REQUIRED)

    # New station grid (required)
    r_R = radial_grid(N_STATIONS, R_ROOT_NORM)

    # =========================
    # Robust initial guess x0
    # =========================
    omega_max = M_TIP_MAX * float(fl.a) / R_required
    omega0 = 0.90 * omega_max

    # Start with moderate-large chord for feasibility (normalized; multiplied by c_ref inside BEMT)
    c_ctrl0 = np.array([0.10, 0.11, 0.10, 0.07], dtype=float)

    # Start with positive pitch; rotor2 slightly reduced
    b1_root0, b1_tip0 = 18.0, 8.0
    b2_root0, b2_tip0 = 16.0, 7.0

    x0 = np.array([ 
        c_ctrl0[0], c_ctrl0[1], c_ctrl0[2], c_ctrl0[3],
        b1_root0, b1_tip0,
        b2_root0, b2_tip0,
        omega0
    ], dtype=float)

    # =========================
    # Bounds
    # =========================
    c_lb, c_ub = 0.03, 0.18
    b_lb, b_ub = -5.0, 45.0
    omega_min = 0.20 * omega_max

    bounds = Bounds(
        [c_lb, c_lb, c_lb, c_lb, b_lb, b_lb, b_lb, b_lb, omega_min],
        [c_ub, c_ub, c_ub, c_ub, b_ub, b_ub, b_ub, b_ub, omega_max]
    )

    # =========================
    # Quick feasibility check
    # =========================
    x_feas = x0.copy()
    x_feas[0:4] = 0.18
    x_feas[4:8] = [35.0, 20.0, 33.0, 18.0]
    x_feas[8] = omega_max

    totals_f, _, _, _, Pf, _, _ = evaluate_system(x_feas, base_rot1, base_rot2, fl, V_CLIMB, r_R)
    print(f"[Feasibility] Aggressive settings -> T_total={totals_f['T_total']:.1f} N, target={T_target_system:.1f} N")

    if totals_f["T_total"] < 0.95 * T_target_system:
        print("[Feasibility WARNING] Even aggressive settings cannot reach target thrust with current model/bounds.")
        print("Increase chord upper bound, pitch bounds, or revisit assumptions if this persists.")
        # Continue anyway.

    # =========================
    # STAGE 1: penalty optimization (Mach only)
    # =========================
    print("\n=== STAGE 1: PENALTY OPTIMIZATION (no thrust equality) ===")

    cons_stage1 = [
        {"type": "ineq", "fun": tip_mach_ineq_constraint, "args": (fl, R_required, M_TIP_MAX)}
    ]

    res1 = minimize(
        fun=objective_stage1_penalty,
        x0=x0,
        args=(base_rot1, base_rot2, fl, V_CLIMB, r_R, T_target_system),
        method="SLSQP",
        bounds=bounds,
        constraints=cons_stage1,
        options={"maxiter": 200, "ftol": 1e-6, "disp": True}
    )

    print("\nStage 1 success:", res1.success)
    print("Stage 1 message:", res1.message)

    x1 = res1.x if res1.success else x0
    totals_1, _, _, _, P1, _, _ = evaluate_system(x1, base_rot1, base_rot2, fl, V_CLIMB, r_R)
    print(f"Stage 1 result: T_total={totals_1['T_total']:.1f} N (target {T_target_system:.1f} N), P_shaft={P1/1000:.2f} kW")

    # =========================
    # STAGE 2: constrained optimization (thrust equality + Mach)
    # =========================
    print("\n=== STAGE 2: CONSTRAINED OPTIMIZATION (thrust equality) ===")

    cons_stage2 = [
        {"type": "eq", "fun": thrust_eq_constraint, "args": (base_rot1, base_rot2, fl, V_CLIMB, r_R, T_target_system)},
        {"type": "ineq", "fun": tip_mach_ineq_constraint, "args": (fl, R_required, M_TIP_MAX)}
    ]

    res2 = minimize(
        fun=objective_power,
        x0=x1,
        args=(base_rot1, base_rot2, fl, V_CLIMB, r_R),
        method="SLSQP",
        bounds=bounds,
        constraints=cons_stage2,
        options={"maxiter": 250, "ftol": 1e-7, "disp": True}
    )

    print("\n=== OPTIMIZATION COMPLETE ===")
    print("Success:", res2.success)
    print("Message:", res2.message)
    print("Final objective P_shaft [W]:", res2.fun)

    if not res2.success:
        raise RuntimeError(f"Optimization failed: {res2.message}")

    x_opt = res2.x

    totals, out1, out2, _, P_shaft, r1_opt, r2_opt = evaluate_system(x_opt, base_rot1, base_rot2, fl, V_CLIMB, r_R)

    omega_opt = float(x_opt[8])
    rpm_opt = omega_opt * 60.0 / (2.0 * np.pi)
    M_tip_opt = omega_opt * R_required / float(fl.a)

    print("\n=== OPTIMUM SUMMARY ===")
    print(f"Radius used (code override): {R_required:.3f} m")
    print(f"c_ref used (code override) : {C_REF_OVERRIDE_M:.3f} m")
    print(f"Omega_opt: {omega_opt:.4f} rad/s  ({rpm_opt:.2f} RPM)")
    print(f"M_tip_opt: {M_tip_opt:.4f} (limit {M_TIP_MAX:.3f})")
    print(f"T_total  : {totals['T_total']:.2f} N (target {T_target_system:.2f} N)")
    print(f"T1, T2   : {totals['T1']:.2f} N , {totals['T2']:.2f} N")
    print(f"Q_total  : {totals['Q_total']:.3f} Nm")
    print(f"P_shaft  : {P_shaft/1000:.2f} kW")

    # Plots (geometry + loads)
    plot_geometry_and_loads(r_R, r1_opt, r2_opt, out1, out2)

    # REQUIRED plot: power vs climb speed with RPM trim to maintain thrust
    plot_power_sweep_trimmed(
        x_opt, base_rot1, base_rot2, fl, r_R,
        V_design=V_CLIMB,
        T_target=T_target_system,
        omega_min=omega_min,
        omega_max=omega_max
    )

    plot_blade_planform(r1_opt, title="Blade planform (optimized geometry) — Rotor 1")
    plot_blade_planform(r2_opt, title="Blade planform (optimized geometry) — Rotor 2")

    # OPTIONAL: save optimized geometry YAML (new file; does not modify original)
    out_yaml = os.path.join("data", "pybemt_optimized_30stations.yaml")
    r1_opt.save_to_yaml(out_yaml)
    print(f"\nSaved optimized Rotor 1 to: {out_yaml}")


if __name__ == "__main__":
    main()
