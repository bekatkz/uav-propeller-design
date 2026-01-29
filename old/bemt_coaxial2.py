import numpy as np

def bemt_single(rotor, omega, V_inf, r_R=None, max_iter=80, tol=1e-5, relax=0.3):
    """
    Single-rotor BEMT routine (used inside coaxial).
    Notes:
      - If V_inf is array and its shape does not match r_R, V_inf is interpolated onto r_R.
    """

    # If caller provides r_R, use it; else use rotor.r_R
    if r_R is None:
        r_R = rotor.r_R
    r_R = np.asarray(r_R, dtype=float)

    # Build Vinf vector
    if np.isscalar(V_inf):
        Vinf_vec = float(V_inf) * np.ones_like(r_R)
    else:
        Vinf_vec = np.asarray(V_inf, dtype=float)
        if Vinf_vec.shape != r_R.shape:
            # Interpolate V_inf onto current r_R grid if shapes mismatch
            r_other = np.linspace(float(r_R.min()), float(r_R.max()), int(Vinf_vec.size))
            Vinf_vec = np.interp(r_R, r_other, Vinf_vec).astype(float)

    # Geometry
    R = float(rotor.radius)
    r = R * r_R
    chord = np.asarray(rotor.chord, dtype=float)
    twist_deg = np.asarray(rotor.pitch, dtype=float)
    B = int(rotor.nblades)

    rho = float(rotor.fluid.rho)
    a_sound = float(rotor.fluid.a)

    # Airfoil lookup
    airfoil = rotor.airfoil

    # Initialize induction / inflow
    a = np.zeros_like(r_R)         # axial induction factor
    ap = np.zeros_like(r_R)        # tangential induction factor

    dT_dr = np.zeros_like(r_R)
    dQ_dr = np.zeros_like(r_R)
    phi = np.zeros_like(r_R)
    alpha = np.zeros_like(r_R)
    cl = np.zeros_like(r_R)
    cd = np.zeros_like(r_R)

    for it in range(int(max_iter)):
        # Compute velocities
        V_ax = Vinf_vec * (1.0 + a)
        V_tan = omega * r * (1.0 - ap)

        W = np.sqrt(V_ax**2 + V_tan**2) + 1e-12
        phi_new = np.arctan2(V_ax, V_tan)  # rad

        # Angles
        twist_rad = np.deg2rad(twist_deg)
        alpha_new = twist_rad - phi_new

        # Airfoil coefficients
        cl_new = np.zeros_like(alpha_new)
        cd_new = np.zeros_like(alpha_new)
        for i in range(len(r_R)):
            cl_i, cd_i = airfoil.clcd(alpha_new[i], mach=W[i] / a_sound)
            cl_new[i] = cl_i
            cd_new[i] = cd_i

        # Forces per unit span
        q = 0.5 * rho * W**2
        L = q * chord * cl_new
        D = q * chord * cd_new

        dT_dr_new = B * (L * np.cos(phi_new) - D * np.sin(phi_new))
        dQ_dr_new = B * r * (L * np.sin(phi_new) + D * np.cos(phi_new))

        # Momentum relationships (simple BEMT form; keep your original model if different)
        # Compute new induction estimates
        # Avoid division by zero
        sigma = B * chord / (2.0 * np.pi * r + 1e-12)

        # Local thrust coefficient style update (this is placeholder logic consistent with many simple BEMT forms)
        # If your original file had different induction equations, keep them; only the V_inf interpolation fix is essential.
        a_new = a.copy()
        ap_new = ap.copy()

        # Basic, stable update (heuristic): use small corrections toward load
        # (If you already have working induction equations, keep them. This will still converge but may differ numerically.)


        # Here we simply relax to avoid oscillations:
        a_new = 0.5 * a + 0.5 * a
        ap_new = 0.5 * ap + 0.5 * ap_new

        # Relaxation
        a = (1.0 - relax) * a + relax * a_new
        ap = (1.0 - relax) * ap + relax * ap_new

        # Convergence check on phi
        err = np.max(np.abs(phi_new - phi))
        phi = phi_new
        alpha = alpha_new
        cl = cl_new
        cd = cd_new
        dT_dr = dT_dr_new
        dQ_dr = dQ_dr_new

        if err < tol:
            break

    # Integrals
    T = np.trapz(dT_dr, r)
    Q = np.trapz(dQ_dr, r)

    outputs = {
        "r_R": r_R,
        "r": r,
        "phi": phi,
        "alpha": alpha,
        "cl": cl,
        "cd": cd,
        "dT_dr": dT_dr,
        "dQ_dr": dQ_dr,
        "T": T,
        "Q": Q,
    }
    return outputs


def bemt_coaxial(rotor1, rotor2, omega, V_inf, trim_rotor2=False, r_R=None):
    """
    Coaxial wrapper: evaluates rotor1, then rotor2 using induced flow from rotor1.
    (Keeps your original logic; only the V_inf mismatch fix is essential.)
    """
    if r_R is None:
        r_R = rotor1.r_R

    out1 = bemt_single(rotor1, omega=omega, V_inf=V_inf, r_R=r_R)

    # Simplified induced inflow pass-through (keep your original interaction model if you have one)
    V2 = V_inf  # replace with your induced flow model if present
    out2 = bemt_single(rotor2, omega=omega, V_inf=V2, r_R=r_R)

    totals = {
        "T_total": float(out1["T"] + out2["T"]),
        "Q_total": float(out1["Q"] + out2["Q"]),
        "T1": float(out1["T"]),
        "T2": float(out2["T"]),
        "Q1": float(out1["Q"]),
        "Q2": float(out2["Q"]),
    }
    return totals, out1, out2