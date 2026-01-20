import os
import numpy as np
import matplotlib.pyplot as plt

import fluid
import propeller
from access_clcd import get_CL_CD_from_neuralfoil


def bemt_single(prop: propeller.Propeller, fl: fluid.Fluid, V_inf, omega,
                R_cutout_frac=0.15, tol=1e-5, max_iter=300, relax=0.2):
    """
    BEMT solver that supports:
      - V_inf as scalar OR array of length len(r_R) (radially varying inflow)
      - returns distributions needed for coaxial coupling and plotting
    """
    radius = float(prop.radius)
    r_R = np.asarray(prop.r_R, dtype=float)
    r = r_R * radius

    chord = np.asarray(prop.chord, dtype=float) * float(prop.c_ref)
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
            raise ValueError("If V_inf is an array, it must have same length as prop.r_R")

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
        if r[i] < R_cutout or r[i] <= 0.0:
            continue
        if abs(r[i] - radius) < 1e-12:
            continue

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
            inside = F * (Vinf_i**2 * F + Cn * sigma_l * U**2)
            root_term = np.sqrt(abs(inside))
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
    return propeller.Propeller(
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


def build_rotor2_inflow(rot1_out, rot1_radius, rot2, V_inf):
    r1 = rot1_out["r"]
    vi1 = rot1_out["vi"]

    r2 = np.asarray(rot2.r_R, dtype=float) * float(rot2.radius)
    vi1_on_r2 = np.interp(r2, r1, vi1, left=vi1[0], right=vi1[-1])

    inside = r2 <= float(rot1_radius) + 1e-12

    # Fully developed slipstream assumption (your notes): Vwake = Vinf + 2*vi
    Vinf2 = np.where(inside, float(V_inf) + 2.0 * vi1_on_r2, float(V_inf))

    coupling = {
        "r2": r2,
        "inside_slipstream": inside,
        "vi1_on_r2": vi1_on_r2,
        "Vinf2": Vinf2
    }
    return Vinf2, coupling


def trim_rotor2_pitch(rot2_base: propeller.Propeller, fl: fluid.Fluid, Vinf2, omega2,
                      alpha_target_deg=2.0, trim_span=(0.30, 0.90),
                      bounds_deg=(-5.0, 20.0), max_iter=18):
    """
    Adjust a uniform pitch offset on rotor 2 so that mean alpha over trim_span hits alpha_target_deg.
    Uses bisection on mean(alpha)-alpha_target.
    """
    rR = np.asarray(rot2_base.r_R, dtype=float)

    def mean_alpha_for_offset(dtheta):
        rot2 = apply_pitch_offset(rot2_base, dtheta)
        _, _, out2 = bemt_single(rot2, fl, V_inf=Vinf2, omega=omega2)
        mask = (rR >= trim_span[0]) & (rR <= trim_span[1])
        # Exclude zeros from cutout if present
        alpha_vals = out2["alpha"][mask]
        return float(np.mean(alpha_vals)), out2

    lo, hi = bounds_deg
    alpha_lo, out_lo = mean_alpha_for_offset(lo)
    alpha_hi, out_hi = mean_alpha_for_offset(hi)

    f_lo = alpha_lo - alpha_target_deg
    f_hi = alpha_hi - alpha_target_deg

    # If no sign change, pick whichever is closer (still improves stability vs no trim)
    if f_lo * f_hi > 0:
        if abs(f_lo) < abs(f_hi):
            return lo, out_lo
        else:
            return hi, out_hi

    out_mid = None
    mid = None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        alpha_mid, out_mid = mean_alpha_for_offset(mid)
        f_mid = alpha_mid - alpha_target_deg

        if abs(f_mid) < 0.05:  # good enough
            break

        if f_lo * f_mid <= 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid

    return float(mid), out_mid


def coaxial_bemt_fixed(rot1, rot2, fl, V_inf, omega1, omega2, trim_rotor2=True, alpha_target_deg=2.0):

    """
    Coaxial model with:
      - Adjustable wake strength k_wake (default 1.0)
      - Optional rotor-2 pitch trim to avoid negative AoA / windmilling
    """
    # Rotor 1
    T1, Q1, out1 = bemt_single(rot1, fl, V_inf=V_inf, omega=omega1)

    # Build rotor-2 inflow (top-hat slipstream)
    Vinf2, coupling = build_rotor2_inflow(out1, rot1.radius, rot2, V_inf)


    # Rotor 2 (optionally trimmed)
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

    plt.figure()
    plt.plot(rR1, out1["Vax"], label="Rotor 1: Vax = Vinf + vi")
    plt.plot(rR2, out2["Vax"], label="Rotor 2: Vax = Vinf2 + vi")
    plt.xlabel("r/R")
    plt.ylabel("Axial speed at disk Vax [m/s]")
    plt.title(f"Disk axial inflow speed (coaxial){title_suffix}")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(rR1, out1["alpha"], label="Rotor 1: alpha")
    plt.plot(rR2, out2["alpha"], label="Rotor 2: alpha")
    plt.xlabel("r/R")
    plt.ylabel("Angle of attack alpha [deg]")
    plt.title(f"Angle of attack distribution (coaxial){title_suffix}")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(rR2, coupling["Vinf2"], label="Rotor 2 inflow Vinf2(r)")
    plt.xlabel("r/R (Rotor 2)")
    plt.ylabel("Imposed inflow Vinf2 [m/s]")
    plt.title(f"Imposed inflow to Rotor 2 (top-hat slipstream){title_suffix}")
    plt.grid(True)
    plt.legend()

    print("\n=== COAXIAL RESULTS (FIXED) ===")
    print(f"Rotor 2 pitch offset = {totals['rotor2_pitch_offset_deg']:.2f} deg")
    print(f"T1 = {totals['T1']:.3f} N, Q1 = {totals['Q1']:.5f} Nm")
    print(f"T2 = {totals['T2']:.3f} N, Q2 = {totals['Q2']:.5f} Nm")
    print(f"T_total = {totals['T_total']:.3f} N, Q_total = {totals['Q_total']:.5f} Nm")

    plt.show()


def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    yaml1 = os.path.join(BASE_DIR, "data", "pybemt_tmotor28.yaml")
    yaml2 = os.path.join(BASE_DIR, "data", "pybemt_tmotor28.yaml")

    rot1 = propeller.Propeller.load_from_yaml(yaml1)
    rot2 = propeller.Propeller.load_from_yaml(yaml2)
    rot2.radius = 0.9* rot1.radius

    fl = fluid.Fluid(altitude=0.0)

    omega = 5000.0 * 2.0 * np.pi / 60.0
    V_inf = 5.0

    # Key knob:
    # - k_wake=1.0 is a practical “near-wake at rotor-2 plane” approximation and avoids over-blowing rotor 2.
    # - If you truly want “fully developed” far wake, set k_wake=2.0, but then trimming rotor 2 is mandatory.
    totals, out1, out2, coupling = coaxial_bemt_fixed(
        rot1, rot2, fl,
        V_inf=V_inf,
        omega1=omega,
        omega2=omega,
        trim_rotor2=True,
        alpha_target_deg=2.0
    )

    plot_diagnostics(out1, out2, totals, coupling, title_suffix=f" | V_inf={V_inf}, omega={omega:.1f}")


if __name__ == "__main__":
    main()
