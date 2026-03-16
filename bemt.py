# -*- coding: utf-8 -*-
"""
=============================================================================
BLADE ELEMENT MOMENTUM THEORY (BEMT) SOLVER - COAXIAL ROTORS
=============================================================================
Description:
    This module evaluates the aerodynamic performance of a coaxial rotor system.
    It balances the local aerodynamic forces on the blade (Blade Element Theory) 
    with the macroscopic acceleration of air through the rotor disk (Momentum Theory).

Theory & Design Choices:
    1. Prandtl Tip/Root Loss: Accounts for the pressure equalization (vortices) 
       at the physical ends of the finite blades.
    2. Coaxial Wake Interference: Uses a one-way far-wake approximation. The 
       upper rotor accelerates the air, contracting the slipstream and acting 
       as an increased "effective freestream" for the lower rotor.
    3. Numerical Stability: Uses under-relaxation to prevent the iterative 
       induction solvers from oscillating or diverging.
=============================================================================
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple

# Import the cached airfoil lookup table (NeuralFoil/NACA database)
from access_clcd import get_CL_CD_array


# =============================================================================
# 1. MATH & GEOMETRY UTILITIES
# =============================================================================

def _safe_clip(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Helper to bound numpy arrays to prevent physics engine crashes."""
    return np.minimum(np.maximum(x, lo), hi)

def _default_radial_grid(R: float, n_stations: int, r_root_cutout: float) -> Tuple[np.ndarray, np.ndarray]:
    """Generates the radial evaluation nodes, avoiding the exact singularity at the root."""
    r_root = r_root_cutout * R
    r = np.linspace(r_root + 0.005 * R, R * 0.995, int(n_stations))
    r_R = r / R
    return r, r_R


# =============================================================================
# 2. AERODYNAMIC LOSS MODELS
# =============================================================================

def prandtl_loss(B: int, r: np.ndarray, R: float, r_root: float, phi: np.ndarray) -> np.ndarray:
    """
    Theory: 
        Standard Momentum Theory assumes a rotor is an infinite disk with an 
        infinite number of blades. The Prandtl loss factor (F) corrects this 
        by modeling the lift drop-off caused by air spilling over the finite 
        blade tips and the root cutout (tip and root vortices).
    """
    eps = 1e-9
    sinphi = np.maximum(np.abs(np.sin(phi)), eps)

    # Tip loss (approaches 0 at r = R)
    f_tip = (B / 2.0) * (R - r) / (r * sinphi)
    f_tip = np.clip(f_tip, 0.0, 50.0)
    F_tip = (2.0 / np.pi) * np.arccos(np.exp(-f_tip))

    # Root loss (approaches 0 at r = r_root)
    r_eps = 1e-4 * R                     
    dr = np.maximum(r - r_root, 0.0)     
    f_root = (B / 2.0) * (dr + r_eps) / (r * sinphi)
    f_root = np.clip(f_root, 0.0, 50.0)
    F_root = (2.0 / np.pi) * np.arccos(np.exp(-f_root))

    # Total combined loss
    F = F_tip * F_root
    return np.clip(F, 1e-4, 1.0)


# =============================================================================
# 3. SINGLE ROTOR BEMT SOLVER
# =============================================================================

def bemt_single(
    *,
    rho: float, mu: float, a_sound: float,
    T_target: float, R: float, B: int, omega: float,
    chord: np.ndarray, twist_deg: np.ndarray, V_inf: float,
    airfoil: int = 2412, r_R: Optional[np.ndarray] = None,
    n_stations: int = 30, r_root_cutout: float = 0.1,
    max_iter: int = 200, tol: float = 1e-4, relax: float = 0.35,
) -> Dict[str, np.ndarray]:
    """
    Solves BEMT for an isolated rotor in axial flow using an iterative approach.
    """
    
    # 1. Setup Grid & Interpolate Geometry
    if r_R is None:
        r, r_R_used = _default_radial_grid(R, n_stations, r_root_cutout)
    else:
        r_R_used = np.asarray(r_R, dtype=float).reshape(-1)
        r_R_used = r_R_used[r_R_used >= r_root_cutout - 1e-12]
        r = r_R_used * R

    chord = np.asarray(chord, dtype=float).reshape(-1)
    twist_deg = np.asarray(twist_deg, dtype=float).reshape(-1)

    # 2. Precompute Kinematics
    dr = np.gradient(r)
    Vt0 = np.maximum(omega * r, 1e-6)  # Base tangential velocity

    vi = np.full_like(r, 2.0)         # [m/s] Initial guess for induced velocity
    ap = np.zeros_like(r)             # Initial guess for swirl induction

    r_root_m = r_root_cutout * R

    # Output arrays
    phi, alpha, cl, cd = np.zeros_like(r), np.zeros_like(r), np.zeros_like(r), np.zeros_like(r)
    dTdr, dQdr, Vax = np.zeros_like(r), np.zeros_like(r), np.zeros_like(r)

    # 3. Iterative Solver Loop
    for _ in range(max_iter):
        vi_old, ap_old = vi.copy(), ap.copy()

        # Step A: Local Velocities & Flow Angles
        Vt = Vt0 * (1.0 - ap)                  # Effective tangential velocity
        Vax = V_inf + vi                       # Effective axial velocity
        W = np.maximum(np.sqrt(Vt**2 + Vax**2), 1e-6) # Local relative wind

        phi = np.arctan2(Vax, Vt)         # Inflow angle
        alpha = np.deg2rad(twist_deg) - phi # Effective Angle of Attack
        alpha_deg = np.rad2deg(alpha)

        # Step B: Aerodynamics
        Re = rho * W * chord / np.maximum(mu, 1e-12)
        Ma = W / np.maximum(a_sound, 1e-9)
        cl, cd = get_CL_CD_array(airfoil, alpha_deg, Re, Ma)

        # Step C: Forces per unit span
        q = 0.5 * rho * W**2
        Lp = q * chord * cl
        Dp = q * chord * cd

        # Step D: Apply Prandtl Losses
        F = prandtl_loss(B, r, R, r_root_m, phi)
        
        # Sectional Thrust and Torque
        dTdr_eff = B * (Lp * np.cos(phi) - Dp * np.sin(phi)) * F
        dQdr_eff = B * r * (Lp * np.sin(phi) + Dp * np.cos(phi)) * F
        
        # Step E: Momentum Update (Solve for vi_new using annulus momentum)
        denom_T = 4.0 * np.pi * rho * r * F
        K = dTdr_eff / np.maximum(denom_T, 1e-12)
        disc = np.maximum(V_inf**2 + 4.0 * K, 0.0)
        vi_new = 0.5 * (-V_inf + np.sqrt(disc))
        
        # Swirl update
        denom_Q = 4.0 * np.pi * rho * (r**3) * F * np.maximum(np.abs(Vax), 1e-6)
        vtheta_new = dQdr_eff / np.maximum(denom_Q, 1e-12)
        ap_new = vtheta_new / Vt0
        
        # Robustness clipping
        vi_new = np.clip(vi_new, 0.0, 0.8 * Vt0)
        ap_new = np.clip(ap_new, -0.6, 0.6)

        # IMPORTANT BUG FIX: Store outputs BEFORE the break!
        dTdr, dQdr = dTdr_eff, dQdr_eff

        # Step F: Under-Relaxation
        vi = (1.0 - relax) * vi + relax * vi_new
        ap = (1.0 - relax) * ap + relax * ap_new

        # Check Convergence
        if max(np.max(np.abs(vi - vi_old)), np.max(np.abs(ap - ap_old))) < tol:
            break

    # 4. Final Integrations
    T = float(np.trapz(dTdr, r))
    Q = float(np.trapz(dQdr, r))
    P = float(Q * omega)

    return {
        "r": r, "r_R": r_R_used, "chord": chord, "twist_deg": twist_deg,
        "phi": phi, "alpha_deg": np.rad2deg(alpha), "CL": cl, "CD": cd,
        "Vax": Vax, "dTdr": dTdr, "dQdr": dQdr,
        "T": np.array([T]), "Q": np.array([Q]), "P": np.array([P]),
    }


# =============================================================================
# 4. COAXIAL ROTOR WRAPPER (WAKE INTERFERENCE)
# =============================================================================

def coaxial_bemt_fixed(
    *,
    rho: float, mu: float, a_sound: float,
    T_total_target: float, R: float, B: int, omega: float,
    chord_upper: np.ndarray, chord_lower: np.ndarray,
    twist_upper_deg: np.ndarray, twist_lower_deg: np.ndarray,
    V_inf: float, airfoil: int = 2412, spacing: float = 0.2,          
    wake_factor: float = 2.0, n_stations: int = 30, r_root_cutout: float = 0.1,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Theory:
        Simulates two rotors stacked coaxially. It calculates the upper rotor 
        in 'clean air', determines the induced downward velocity (wake), and adds
        that wake to the freestream of the lower rotor.
    """
    T_each = 0.5 * float(T_total_target)
    
    # Calculate wake strength and decay based on rotor spacing
    decay = 1.0 / (1.0 + float(spacing) / float(R))**2
    effective_wake = float(wake_factor) * decay
    
    if float(V_inf) >= 0.0:
        # HOVER & CLIMB: Upper rotor sees clean air, lower rotor operates in wake
        upper = bemt_single(
            rho=rho, mu=mu, a_sound=a_sound, T_target=T_each, R=R, B=B, omega=omega,
            chord=chord_upper, twist_deg=twist_upper_deg, V_inf=V_inf,
            airfoil=airfoil, n_stations=n_stations, r_root_cutout=r_root_cutout
        )
        
        # Calculate thrust-weighted average induced velocity from upper disk
        vi_upper = upper["Vax"] - float(V_inf)
        T_weight = np.maximum(upper["dTdr"], 0.0)
        vi_upper_mean = (np.average(vi_upper, weights=T_weight)
                         if np.sum(T_weight) > 0 else float(np.mean(vi_upper)))
        
        V_inf_lower = float(V_inf) + effective_wake * vi_upper_mean
        
        lower = bemt_single(
            rho=rho, mu=mu, a_sound=a_sound, T_target=T_each, R=R, B=B, omega=omega,
            chord=chord_lower, twist_deg=twist_lower_deg, V_inf=V_inf_lower,
            airfoil=airfoil, n_stations=n_stations, r_root_cutout=r_root_cutout
        )

    else:
        # DESCENT: Lower rotor sees clean air, upper rotor operates in the wake
        lower = bemt_single(
            rho=rho, mu=mu, a_sound=a_sound, T_target=T_each, R=R, B=B, omega=omega,
            chord=chord_lower, twist_deg=twist_lower_deg, V_inf=V_inf,
            airfoil=airfoil, n_stations=n_stations, r_root_cutout=r_root_cutout
        )
        
        vi_lower = lower["Vax"] - float(V_inf)
        T_weight = np.maximum(lower["dTdr"], 0.0)
        vi_lower_mean = (np.average(vi_lower, weights=T_weight)
                         if np.sum(T_weight) > 0 else float(np.mean(vi_lower)))
        
        V_inf_upper = float(V_inf) + effective_wake * vi_lower_mean
        
        upper = bemt_single(
            rho=rho, mu=mu, a_sound=a_sound, T_target=T_each, R=R, B=B, omega=omega,
            chord=chord_upper, twist_deg=twist_upper_deg, V_inf=V_inf_upper,
            airfoil=airfoil, n_stations=n_stations, r_root_cutout=r_root_cutout
        )

    # Combine totals
    T_tot = float(upper["T"][0] + lower["T"][0])
    P_tot = float(upper["P"][0] + lower["P"][0])

    return {
        "upper": upper,
        "lower": lower,
        "totals": {
            "T": np.array([T_tot]),
            "P": np.array([P_tot]),
        }
    }