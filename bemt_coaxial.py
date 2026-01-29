# bemt_coaxial.py
import numpy as np
from access_clcd import get_CL_CD_from_neuralfoil

def prandtl_loss(B, r, R, r_root, phi):
    """
    Prandtl tip and root loss correction.
    """
    sphi = np.maximum(np.abs(np.sin(phi)), 1e-6)
    f_tip = (B / 2.0) * (R - r) / (r * sphi + 1e-12)
    f_root = (B / 2.0) * (r - r_root) / (r * sphi + 1e-12)
    
    F_tip = (2.0 / np.pi) * np.arccos(np.exp(-np.clip(f_tip, 0.0, 50.0)))
    F_root = (2.0 / np.pi) * np.arccos(np.exp(-np.clip(f_root, 0.0, 50.0)))
    
    return np.clip(F_tip * F_root, 1e-4, 1.0)

def bemt_single(rotor, fluid, V_inf, omega, r_R=None, max_iter=100, tol=1e-5, relax=0.25):
    """
    Single-rotor BEMT (axial flow).
    """
    if r_R is None: r_R = rotor.r_R
    r_R = np.asarray(r_R, dtype=float)
    R, B = float(rotor.radius), int(rotor.nblades)
    rho, nu, a_sound = float(fluid.rho), float(fluid.nu), float(fluid.a)
    
    # Geometry
    r = R * r_R
    chord = np.asarray(rotor.chord, dtype=float) * float(rotor.c_ref)
    pitch_rad = np.radians(np.asarray(rotor.pitch, dtype=float))
    airfoil_ids = np.asarray(rotor.airfoil, dtype=int)
    
    # Inflow handling (scalar or vector)
    if np.isscalar(V_inf): 
        V_inf = np.full_like(r_R, V_inf)
    
    # Initialize induction factors
    a = np.full_like(r_R, 0.1) 
    ap = np.full_like(r_R, 0.01)

    phi, alpha, dT_dr, dQ_dr = [np.zeros_like(r_R) for _ in range(4)]
    Vax, Vtan = np.zeros_like(r_R), np.zeros_like(r_R)

    for _ in range(int(max_iter)):
        # 1. Velocities
        Vax = V_inf * (1.0 + a)
        Vtan = omega * r * (1.0 - ap)
        W = np.sqrt(Vax**2 + Vtan**2)
        phi = np.arctan2(Vax, Vtan)
        
        # 2. Aerodynamics
        alpha = pitch_rad - phi
        cl, cd = np.zeros_like(alpha), np.zeros_like(alpha)
        
        for i in range(len(r)):
            Re = W[i] * chord[i] / nu
            Ma = W[i] / a_sound
            
            # NeuralFoil access
            # Note: alpha is in radians, function expects degrees if not specified otherwise
            # Checking typical access_clcd usage, usually expects degrees.
            cl[i], cd[i] = get_CL_CD_from_neuralfoil(int(airfoil_ids[i]), np.degrees(alpha[i]), Re, Ma)
            
        Cn = cl * np.cos(phi) - cd * np.sin(phi)
        Ct = cl * np.sin(phi) + cd * np.cos(phi)
        
        # 3. BEMT Updates
        F = prandtl_loss(B, r, R, r[0], phi)
        sigma = B * chord / (2.0 * np.pi * r)
        
        # Thrust Coefficient Term
        denom_a = (4.0 * F * np.sin(phi)**2) / (sigma * Cn + 1e-12) - 1.0
        # Torque Coefficient Term
        denom_ap = (4.0 * F * np.sin(phi) * np.cos(phi)) / (sigma * Ct + 1e-12) + 1.0
        
        a_new = 1.0 / denom_a
        ap_new = 1.0 / denom_ap
        
        # Clip to prevent divergence
        a_new = np.clip(a_new, -0.5, 10.0)
        ap_new = np.clip(ap_new, -0.5, 0.5)
        
        # Relaxation
        a = (1.0 - relax) * a + relax * a_new
        ap = (1.0 - relax) * ap + relax * ap_new
        
        if np.max(np.abs(a_new - a)) < tol: 
            break
            
    # Final Loads
    dT_dr = 0.5 * rho * W**2 * chord * B * Cn
    dQ_dr = 0.5 * rho * W**2 * chord * B * Ct * r
    
    return {
        "r": r, 
        "T": np.trapz(dT_dr, r), 
        "Q": np.trapz(dQ_dr, r),
        "Vax": Vax, 
        "dT_dr": dT_dr, 
        "phi": phi, 
        "alpha": alpha
    }

def coaxial_bemt_fixed(rotor1, rotor2, fluid, V_inf, omega1, omega2, r_R, max_coupling_iter=15, **kwargs):
    """
    Coaxial BEMT with Stream Funnel (Contraction) Model.
    
    1. Solves Rotor 1 (Upper).
    2. Contracts the wake radius by 1/sqrt(2) ~ 0.707 (Fully developed slipstream).
    3. Maps the wake onto Rotor 2 (Lower).
    4. Solves Rotor 2 with the mixed inflow.
    """
    # Prepare Physical Grids
    R1, R2 = float(rotor1.radius), float(rotor2.radius)
    r1_phys = r_R * R1
    r2_phys = r_R * R2
    
    # --- FIXED: STREAM FUNNEL CONTRACTION FACTOR ---
    # For a fully developed slipstream, Area_inf = Area_disk / 2
    # Therefore, Radius_inf = Radius_disk / sqrt(2)
    CONTRACTION_FACTOR = 1.0 / np.sqrt(2.0)  # Approx 0.7071
    
    out1, out2 = None, None
    
    for _ in range(max_coupling_iter):
        # 1. Evaluate Top Rotor
        out1 = bemt_single(rotor1, fluid, V_inf, omega1, r_R)
        
        # 2. Map Wake from R1 to R2 (WITH CONTRACTION)
        # Calculate induced velocity at R1 disk
        vi1 = out1["Vax"] - V_inf 
        
        # Fully developed wake velocity = 2 * vi1
        wake_vel_far = 2.0 * vi1
        
        # Contract the radial coordinates of the wake
        # The wake profile that existed at r1_phys is now squeezed into r_wake_phys
        r_wake_phys = r1_phys * CONTRACTION_FACTOR
        
        # Interpolate this contracted wake onto Rotor 2's physical coordinates.
        # Points on Rotor 2 that are OUTSIDE the contracted wake (r > R1*0.707) 
        # will get 0.0 (left=0.0, right=0.0), i.e., undisturbed inflow.
        wake_on_R2 = np.interp(r2_phys, r_wake_phys, wake_vel_far, left=0.0, right=0.0)
        
        # 3. Evaluate Bottom Rotor
        V2_inflow = V_inf + wake_on_R2
        out2 = bemt_single(rotor2, fluid, V2_inflow, omega2, r_R)
        
    totals = {
        "T1": out1["T"], "Q1": out1["Q"],
        "T2": out2["T"], "Q2": out2["Q"],
        "T_total": out1["T"] + out2["T"],
        "Q_total": out1["Q"] + out2["Q"]
    }
    return totals, out1, out2, {"coupling_iters": max_coupling_iter}