# bemt_coaxial.py
import numpy as np

from access_clcd import get_CL_CD_from_neuralfoil


def prandtl_loss(B, r, R, r_root, phi):
    """
    Tip + root loss factor F in [0..1].
    F ~ 1 mid-span, F -> 0 near tip/root.
    """
    sphi = np.maximum(np.abs(np.sin(phi)), 1e-6)

    f_tip = (B / 2.0) * (R - r) / (r * sphi + 1e-12)
    f_root = (B / 2.0) * (r - r_root) / (r * sphi + 1e-12)

    F_tip = (2.0 / np.pi) * np.arccos(np.exp(-np.clip(f_tip, 0.0, 50.0)))
    F_root = (2.0 / np.pi) * np.arccos(np.exp(-np.clip(f_root, 0.0, 50.0)))
    F = F_tip * F_root
    return np.clip(F, 1e-3, 1.0)


def bemt_single(
    rotor,
    fluid,
    V_inf,
    omega,
    r_R=None,
    max_iter=80,
    tol=1e-5,
    relax=0.35,
):
    """
    Single-rotor BEMT (axial flow).
    Returns distributions (phi, alpha, dT/dr, dQ/dr, etc.) and totals T, Q.

    Expects:
      rotor.radius, rotor.nblades, rotor.c_ref,
      rotor.r_R, rotor.chord (normalized), rotor.pitch (deg),
      rotor.airfoil (int ids per station)
      fluid.rho, fluid.a, fluid.nu
    """
    if r_R is None:
        r_R = rotor.r_R
    r_R = np.asarray(r_R, dtype=float)

    R = float(rotor.radius)
    r = R * r_R
    r_root = float(R * r_R.min())

    B = int(rotor.nblades)
    rho = float(fluid.rho)
    a_sound = float(fluid.a)
    nu = float(fluid.nu)

    chord = np.asarray(rotor.chord, dtype=float) * float(rotor.c_ref)  # [m]
    pitch_deg = np.asarray(rotor.pitch, dtype=float)                   # [deg]
    airfoil_ids = np.asarray(rotor.airfoil, dtype=int)                 # int per station

    # V_inf can be scalar or array
    if np.isscalar(V_inf):
        Vinf_vec = float(V_inf) * np.ones_like(r_R)
    else:
        Vinf_vec = np.asarray(V_inf, dtype=float)
        if Vinf_vec.shape != r_R.shape:
            rr = np.linspace(float(r_R.min()), float(r_R.max()), int(Vinf_vec.size))
            Vinf_vec = np.interp(r_R, rr, Vinf_vec)

    # Prevent singular behavior at hover-like low inflow
    V_eps = 0.25
    Vinf_eff = np.where(np.abs(Vinf_vec) < V_eps,
                        np.sign(Vinf_vec) * V_eps + (Vinf_vec == 0.0) * V_eps,
                        Vinf_vec)

    # Induction initial guesses
    a = 0.05 * np.ones_like(r_R)
    ap = 0.01 * np.ones_like(r_R)

    # Outputs
    phi = np.zeros_like(r_R)
    alpha = np.zeros_like(r_R)
    dT_dr = np.zeros_like(r_R)
    dQ_dr = np.zeros_like(r_R)
    Vax = np.zeros_like(r_R)
    Vtan = np.zeros_like(r_R)
    W = np.zeros_like(r_R)

    for _ in range(int(max_iter)):
        Vax_new = Vinf_eff * (1.0 + a)
        Vtan_new = omega * r * (1.0 - ap)
        W_new = np.sqrt(Vax_new**2 + Vtan_new**2) + 1e-12
        phi_new = np.arctan2(Vax_new, Vtan_new)

        pitch = np.deg2rad(pitch_deg)
        alpha_new = pitch - phi_new

        # CL/CD from NeuralFoil accessor
        cl = np.zeros_like(alpha_new)
        cd = np.zeros_like(alpha_new)
        for i in range(len(r_R)):
            Re = max(W_new[i] * chord[i] / nu, 1e4)
            Ma = W_new[i] / a_sound
            cl[i], cd[i] = get_CL_CD_from_neuralfoil(int(airfoil_ids[i]), float(alpha_new[i]), float(Re), float(Ma))

        # Normal/tangential coefficients
        Cn = cl * np.cos(phi_new) - cd * np.sin(phi_new)
        Ct = cl * np.sin(phi_new) + cd * np.cos(phi_new)

        # Tip/root loss
        F = prandtl_loss(B, r, R, r_root, phi_new)

        # Solidity
        sigma = B * chord / (2.0 * np.pi * r + 1e-12)

        # Fixed-point update for a and a'
        denom_a = (4.0 * F * (np.sin(phi_new) ** 2)) / (sigma * (Cn + 1e-12)) - 1.0
        denom_ap = (4.0 * F * np.sin(phi_new) * np.cos(phi_new)) / (sigma * (Ct + 1e-12)) + 1.0

        a_new = 1.0 / np.clip(denom_a, 1e-3, 1e3)
        ap_new = 1.0 / np.clip(denom_ap, 1e-3, 1e3)

        a_new = np.clip(a_new, -0.2, 1.0)
        ap_new = np.clip(ap_new, -0.5, 0.5)

        # Section forces
        q = 0.5 * rho * W_new**2
        L = q * chord * cl
        D = q * chord * cd

        dT_new = B * (L * np.cos(phi_new) - D * np.sin(phi_new))
        dQ_new = B * r * (L * np.sin(phi_new) + D * np.cos(phi_new))

        err = float(np.max(np.abs(phi_new - phi)))

        # Relaxation
        a = (1.0 - relax) * a + relax * a_new
        ap = (1.0 - relax) * ap + relax * ap_new

        # Save
        phi = phi_new
        alpha = alpha_new
        dT_dr = dT_new
        dQ_dr = dQ_new
        Vax = Vax_new
        Vtan = Vtan_new
        W = W_new

        if err < tol:
            break

    T = float(np.trapz(dT_dr, r))
    Q = float(np.trapz(dQ_dr, r))

    return {
        "r_R": r_R, "r": r,
        "phi": phi, "alpha": alpha,
        "Vax": Vax, "Vtan": Vtan, "W": W,
        "dT_dr": dT_dr, "dQ_dr": dQ_dr,
        "T": T, "Q": Q,
    }


def coaxial_bemt_fixed(
    rotor1,
    rotor2,
    fluid,
    V_inf,
    omega1,
    omega2,
    r_R=None,
    max_coupling_iter=25,
    coupling_tol=1e-3,
    trim_rotor2=False,     # accepted for compatibility; not used
    alpha_target_deg=3.0,  # accepted for compatibility; not used
    **kwargs,
):
    """
    Coaxial coupling (simple fully-developed slipstream model):
      - Rotor 1 sees V_inf
      - Rotor 2 sees V_inf + 2*vi1 inside rotor1 disk, and V_inf outside.
    Iterated with relaxation but always capped.
    """
    if r_R is None:
        r_R = rotor1.r_R
    r_R = np.asarray(r_R, dtype=float)

    V2_profile = float(V_inf) * np.ones_like(r_R)

    out1 = None
    out2 = None

    for k in range(int(max_coupling_iter)):
        out1 = bemt_single(rotor1, fluid, V_inf=float(V_inf), omega=float(omega1), r_R=r_R)

        vi1 = out1["Vax"] - float(V_inf)  # induced increment at disk
        dV = 2.0 * vi1                    # fully developed slipstream increment

        # rotor2 points that lie inside rotor1 disk radius
        slip_mask = (r_R * float(rotor2.radius)) <= float(rotor1.radius)

        V2_new = float(V_inf) * np.ones_like(r_R)
        V2_new[slip_mask] = float(V_inf) + dV[slip_mask]

        out2 = bemt_single(rotor2, fluid, V_inf=V2_new, omega=float(omega2), r_R=r_R)

        err = float(np.max(np.abs(V2_new - V2_profile)))
        V2_profile = 0.6 * V2_profile + 0.4 * V2_new

        if err < coupling_tol:
            break

    totals = {
        "T1": float(out1["T"]), "Q1": float(out1["Q"]),
        "T2": float(out2["T"]), "Q2": float(out2["Q"]),
        "T_total": float(out1["T"] + out2["T"]),
        "Q_total": float(out1["Q"] + out2["Q"]),
    }
    coupling = {"V2_profile": V2_profile, "coupling_iters": k + 1}
    return totals, out1, out2, coupling