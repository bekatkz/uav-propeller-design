import os
import numpy as np
import matplotlib.pyplot as plt

import fluid
import propeller
from access_clcd import get_CL_CD_from_neuralfoil


def bemt_single(prop: propeller.Propeller, fl: fluid.Fluid, V_inf, omega,
                R_cutout_frac=0.15, tol=1e-5, max_iter=500, relax=0.25):
    """
    BEMT solver for a single rotor.
    Includes relaxation fix (max_iter=500, relax=0.1) for stability.
    """
    radius = float(prop.radius)
    r_R = np.asarray(prop.r_R, dtype=float)
    r = r_R * radius

    chord = np.asarray(prop.chord, dtype=float) 
    # Note: If prop.chord is normalized c/c_ref, multiply by prop.c_ref. 
    # If using optimization.py which sets physical chord, this line handles it if c_ref=1.
    if hasattr(prop, 'c_ref') and prop.c_ref is not None:
         chord = chord * float(prop.c_ref)

    beta = np.asarray(prop.pitch, dtype=float)           # degrees
    nblades = int(prop.nblades)
    rho = float(fl.rho)
    nu = float(fl.nu)
    a_sos = float(fl.a)
    airfoil = np.asarray(prop.airfoil)

    # Allow scalar or radial array inflow
    if np.ndim(V_inf) == 0:
        Vinf_vec = np.full_like(r_R, float(V_inf), dtype=float)
    else:
        Vinf_vec = np.asarray(V_inf, dtype=float)
        if Vinf_vec.shape != r_R.shape:
            # Interpolate V_inf to current r_R if shapes mismatch
            # (Happens when V_inf comes from a different mesh size in coaxial)
            pass 

    R_cutout = R_cutout_frac * radius

    dT_dr = np.zeros_like(r_R)
    dM_dr = np.zeros_like(r_R)

    # Distributions
    vi_r = np.zeros_like(r_R)          # induced velocity
    Vax_r = np.zeros_like(r_R)         # axial at disk
    phi_r = np.zeros_like(r_R)         # rad
    alpha_r = np.zeros_like(r_R)       # deg
    F_r = np.ones_like(r_R)            

    vi_guess = 3.0  # warm-start       

    for i in range(len(r_R)):
        # Safety checks
        if r[i] < R_cutout or r[i] <= 0.0:
            continue
        
        # Handle scalar vs array V_inf
        if np.ndim(V_inf) == 0:
             Vinf_i = float(V_inf)
        else:
             # simple nearest/direct access if grids match
             Vinf_i = Vinf_vec[i]

        sigma_l = nblades * chord[i] / (2.0 * np.pi * r[i])

        a_prime = 0.01
        v_i = vi_guess

        # Iterate induction
        for _ in range(max_iter):
            V_axial = Vinf_i + v_i
            V_tan = omega * r[i] * (1.0 - a_prime)

            phi = np.arctan2(V_axial, V_tan)
            U = np.sqrt(V_axial**2 + V_tan**2)

            alpha = beta[i] - np.degrees(phi)

            Re = U * chord[i] / nu
            Ma = U / a_sos

            cl, cd = get_CL_CD_from_neuralfoil(int(airfoil[i]), alpha=alpha, Re=Re, Ma=Ma)

            Cn = cl * np.cos(phi) - cd * np.sin(phi)
            Ct = cl * np.sin(phi) + cd * np.cos(phi)

            # Tip loss
            f = (nblades / 2.0) * (radius - r[i]) / (r[i] * max(np.sin(phi), 1e-6))
            f = max(f, 1e-4)
            F = min(2.0 / np.pi * np.arccos(np.exp(-f)), 1.0)

            # Update a'
            kappap = 4.0 * F * np.sin(phi) * np.cos(phi) / max(sigma_l * Ct, 1e-8)
            a_prime_new = 1.0 / (kappap + 1.0)
            a_prime = relax * a_prime_new + (1.0 - relax) * a_prime

            # Update v_i
            # Momentum: 4*F*a*(1-a) = sigma*Cn ... approx v_i relation
            # General BEMT residual approach
            inside = F * (Vinf_i**2 * F + Cn * sigma_l * U**2)
            # Avoid negative sqrt
            if inside < 0: 
                inside = 0
            
            root_term = np.sqrt(inside)
            v_i_new = np.sign(Cn) * (1.0 / (2.0 * F)) * (root_term - Vinf_i * F)

            if abs(v_i_new - v_i) < tol:
                v_i = v_i_new
                break

            v_i = relax * v_i_new + (1.0 - relax) * v_i

        # Store
        vi_r[i] = v_i
        Vax_r[i] = Vinf_i + v_i
        phi_r[i] = phi
        alpha_r[i] = alpha
        F_r[i] = F

        # Loads
        dT_dr[i] = 0.5 * rho * U**2 * chord[i] * nblades * Cn
        dM_dr[i] = 0.5 * rho * U**2 * chord[i] * nblades * Ct * r[i]

        vi_guess = v_i

    Thrust = np.trapezoid(dT_dr, x=r)
    Torque = np.trapezoid(dM_dr, x=r)

    out = {
        "r_R": r_R, "r": r,
        "Vinf": Vinf_vec,
        "vi": vi_r, "Vax": Vax_r,
        "phi": phi_r, "alpha": alpha_r,
        "F": F_r,
        "dT_dr": dT_dr, "dM_dr": dM_dr
    }
    return Thrust, Torque, out


def apply_pitch_offset(prop: propeller.Propeller, dtheta_deg: float):
    """Return a new Propeller object with pitch offset (degrees) applied."""
    p_new = propeller.Propeller(
        nblades=prop.nblades,
        solidity=prop.solidity,
        radius=prop.radius,
        omega=prop.omega,
        r_R=np.array(prop.r_R, dtype=float),
        chord=np.array(prop.chord, dtype=float),
        pitch=np.array(prop.pitch, dtype=float) + float(dtheta_deg),
        sweep=np.array(prop.sweep, dtype=float),
        airfoil=np.array(prop.airfoil, dtype=float),
        c_ref=prop.c_ref
    )
    # Ensure physical chord matches if it was set manually
    if hasattr(prop, 'c_ref') and prop.c_ref == 1.0:
         p_new.c_ref = 1.0
         p_new.chord = np.array(prop.chord, dtype=float)
    return p_new


def build_rotor2_inflow(rot1_out, rot1_radius, rot2, V_inf):
    """
    Calculates the inflow distribution for Rotor 2.
    Implements:
      - Mass Conservation (Slipstream Contraction / "Funnel")
      - Top-Hat profile (Undisturbed outside, Vinf+2vi inside)
    """
    r1 = rot1_out["r"]
    vi1 = rot1_out["vi"]
    
    # 1. Calculate Average Induced Velocity on Rotor 1
    # Weighted by area for mass balance accuracy
    # vi_mean = Integral(vi * 2*pi*r dr) / Area
    vi_mean = np.trapezoid(vi1 * 2 * np.pi * r1, x=r1) / (np.pi * rot1_radius**2)
    
    # 2. Calculate Contracted Wake Radius (Fully Developed Slipstream)
    # Mass conservation: rho * A1 * (V + vi) = rho * A_wake * (V + 2vi)
    # (Approximate using mean velocities for the contraction ratio)
    V_disk = float(V_inf) + vi_mean
    V_wake = float(V_inf) + 2.0 * vi_mean
    
    contraction_ratio = np.sqrt(V_disk / V_wake) if V_wake > 1e-6 else 1.0
    R_wake = rot1_radius * contraction_ratio

    print(f"   [Debug] Wake Contraction: R_disk={rot1_radius:.3f}m -> R_wake={R_wake:.3f}m")

    # 3. Map to Rotor 2
    r2 = np.asarray(rot2.r_R, dtype=float) * float(rot2.radius)
    
    # Interpolate vi1 distribution? 
    # Usually, we assume the profile contracts radially. 
    # Map r2 (inside wake) back to r1 to fetch the corresponding vi.
    # r1_equivalent = r2 / contraction_ratio
    
    vi1_on_r2 = np.zeros_like(r2)
    
    # For points inside the wake, interpolate based on normalized position in wake
    mask_inside = r2 <= R_wake
    if np.any(mask_inside):
        r1_lookup = r2[mask_inside] / contraction_ratio
        vi1_on_r2[mask_inside] = np.interp(r1_lookup, r1, vi1, left=vi1[0], right=vi1[-1])

    # 4. Apply Top-Hat Model
    # Inside wake: V = V_inf + 2 * vi (interpolated)
    # Outside wake: V = V_inf
    Vinf2 = np.where(mask_inside, float(V_inf) + 2.0 * vi1_on_r2, float(V_inf))

    coupling = {
        "r2": r2,
        "inside_slipstream": mask_inside,
        "vi1_on_r2": vi1_on_r2,
        "Vinf2": Vinf2,
        "R_wake": R_wake
    }
    return Vinf2, coupling


def trim_rotor2_pitch(rot2_base: propeller.Propeller, fl: fluid.Fluid, Vinf2, omega2,
                      alpha_target_deg=2.0, trim_span=(0.30, 0.90),
                      bounds_deg=(-5.0, 20.0), max_iter=5):
    """
    Adjust a uniform pitch offset on rotor 2 so that mean alpha over trim_span hits alpha_target_deg.
    """
    rR = np.asarray(rot2_base.r_R, dtype=float)

    def mean_alpha_for_offset(dtheta):
        rot2 = apply_pitch_offset(rot2_base, dtheta)
        _, _, out2 = bemt_single(rot2, fl, V_inf=Vinf2, omega=omega2)
        mask = (rR >= trim_span[0]) & (rR <= trim_span[1])
        alpha_vals = out2["alpha"][mask]
        if len(alpha_vals) == 0: return 0.0, out2
        return float(np.mean(alpha_vals)), out2

    lo, hi = bounds_deg
    alpha_lo, out_lo = mean_alpha_for_offset(lo)
    alpha_hi, out_hi = mean_alpha_for_offset(hi)

    f_lo = alpha_lo - alpha_target_deg
    f_hi = alpha_hi - alpha_target_deg

    if f_lo * f_hi > 0:
        if abs(f_lo) < abs(f_hi):
            return lo, out_lo
        else:
            return hi, out_hi

    mid = lo
    out_mid = out_lo
    
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        alpha_mid, out_mid = mean_alpha_for_offset(mid)
        f_mid = alpha_mid - alpha_target_deg

        if abs(f_mid) < 0.1: 
            break

        if f_lo * f_mid <= 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid

    return float(mid), out_mid


def coaxial_bemt_fixed(rot1, rot2, fl, V_inf, omega1, omega2, trim_rotor2=True, alpha_target_deg=3.0):

    # 1. Solve Rotor 1
    T1, Q1, out1 = bemt_single(rot1, fl, V_inf=V_inf, omega=omega1)

    # 2. Build Inflow for Rotor 2 (with Contraction)
    Vinf2, coupling = build_rotor2_inflow(out1, rot1.radius, rot2, V_inf)

    # 3. Solve Rotor 2 (with Trim)
    dtheta2 = 0.0
    if trim_rotor2:
        dtheta2, out2 = trim_rotor2_pitch(
            rot2_base=rot2, fl=fl, Vinf2=Vinf2, omega2=omega2,
            alpha_target_deg=alpha_target_deg
        )
        rot2_used = apply_pitch_offset(rot2, dtheta2)
        T2, Q2, out2 = bemt_single(rot2_used, fl, V_inf=Vinf2, omega=omega2)
    else:
        T2, Q2, out2 = bemt_single(rot2, fl, V_inf=Vinf2, omega=omega2)

    totals = {
        "T1": T1, "Q1": Q1,
        "T2": T2, "Q2": Q2,
        "T_total": T1 + T2,
        "Q_total": Q1 + Q2,
        "rotor2_pitch_offset_deg": float(dtheta2)
    }
    return totals, out1, out2, coupling


def plot_diagnostics(out1, out2, totals, coupling, title_suffix=""):
    rR1 = out1["r_R"]
    rR2 = out2["r_R"]

    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    
    # Plot 1: Axial Velocities
    axs[0].plot(rR1, out1["Vax"], label="R1 Vax")
    axs[0].plot(rR2, out2["Vax"], label="R2 Vax")
    axs[0].plot(rR2, coupling["Vinf2"], '--', label="R2 Inflow (Wake)")
    axs[0].set_title(f"Velocities {title_suffix}")
    axs[0].set_xlabel("r/R")
    axs[0].set_ylabel("Velocity [m/s]")
    axs[0].grid(True)
    axs[0].legend()

    # Plot 2: Alpha
    axs[1].plot(rR1, out1["alpha"], label="R1 Alpha")
    axs[1].plot(rR2, out2["alpha"], label="R2 Alpha")
    axs[1].set_title("Angle of Attack")
    axs[1].set_xlabel("r/R")
    axs[1].set_ylabel("Alpha [deg]")
    axs[1].grid(True)
    axs[1].legend()

    # Plot 3: Loads (Thrust Distribution)
    axs[2].plot(rR1, out1["dT_dr"], label="R1 dT/dr")
    axs[2].plot(rR2, out2["dT_dr"], label="R2 dT/dr")
    axs[2].set_title("Thrust Distribution")
    axs[2].set_xlabel("r/R")
    axs[2].set_ylabel("dT/dr [N/m]")
    axs[2].grid(True)
    axs[2].legend()

    print("\n=== COAXIAL RESULTS ===")
    print(f"Rotor 2 Pitch Offset: {totals['rotor2_pitch_offset_deg']:.2f} deg")
    print(f"T1: {totals['T1']:.2f} N, T2: {totals['T2']:.2f} N => Total: {totals['T_total']:.2f} N")
    print(f"P1: {totals['Q1']*523:.1f} W, P2: {totals['Q2']*523:.1f} W") # Approx power display

    plt.tight_layout()
    plt.show()

# =============================================================================
# TEST BLOCK (Copy and replace the "pass" block at the bottom of bemt_coaxial.py)
# =============================================================================
if __name__ == "__main__":
    # 1. Setup paths
    # This ensures it finds the file even if you run it from a different folder
    base_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(base_dir, "data", "pybemt_tmotor28.yaml")
    
    # 2. Load Propellers
    print(f"Loading propeller from: {yaml_path}")
    rot1 = propeller.Propeller.load_from_yaml(yaml_path)
    rot2 = propeller.Propeller.load_from_yaml(yaml_path)
    
    # Optional: Make rotor 2 slightly smaller or different to test flexibility
    # rot2.radius = rot1.radius * 0.95 

    # 3. Define Conditions
    fl = fluid.Fluid(altitude=0.0)
    V_inf = 5.0     # Climb speed [m/s]
    RPM = 5000.0
    omega = RPM * 2 * np.pi / 60.0
    
    # 4. Run the Solver
    print(f"Running Coaxial BEMT at {RPM} RPM, V_inf={V_inf} m/s...")
    
    totals, out1, out2, coupling = coaxial_bemt_fixed(
        rot1, rot2, fl, 
        V_inf=V_inf, 
        omega1=omega, 
        omega2=omega,
        trim_rotor2=True,      # This turns on the pitch adjustment logic
        alpha_target_deg=3.0
    )
    
    # 5. Plot Results
    plot_diagnostics(out1, out2, totals, coupling, title_suffix=" (Test Run)")