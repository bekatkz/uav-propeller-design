# bemt_coaxial.py
"""
Coaxial BEMT model (upper + lower rotor) with one-way coupling:
Upper solved first in freestream, lower sees increased axial inflow due to upper wake.

NOTE:
- This file includes bemt_single() and coaxial_bemt_fixed().
- coaxial_bemt_fixed() now supports independent chord distributions:
    chord_upper, chord_lower
  and keeps backward compatibility with the old single "chord" argument.
"""

from __future__ import annotations

from typing import Dict
import numpy as np


def prandtl_tip_loss(B: int, r: np.ndarray, R: float, phi: np.ndarray) -> np.ndarray:
    """Prandtl tip-loss factor (tip only)."""
    sphi = np.maximum(np.sin(phi), 1e-8)
    f = (B / 2.0) * (R - r) / (r * sphi)
    F = (2.0 / np.pi) * np.arccos(np.clip(np.exp(-f), 0.0, 1.0))
    return np.clip(F, 1e-3, 1.0)


def bemt_single(
    *,
    rho: float,
    mu: float,
    a_sound: float,
    T_target: float,
    R: float,
    B: int,
    omega: float,
    chord: np.ndarray,
    twist_deg: np.ndarray,
    V_inf: float,
    airfoil: int = 2412,
    n_stations: int = 30,
    r_root_cutout: float = 0.1,
    relax: float = 0.25,
    max_iter: int = 250,
    tol: float = 1e-5,
) -> Dict[str, np.ndarray]:
    """
    Single-rotor BEMT (your existing implementation style).

    IMPORTANT:
    This is your original solver structure; it still uses the same induction approach
    you had before. If you want the physically improved induction / swirl usage,
    we can replace bemt_single() too — but here I’m only changing coaxial chord plumbing
    to keep your pipeline stable.
    """
    from access_clcd import get_CL_CD  # keep import local (fast enough with caching)

    # Radial grid
    r = np.linspace(r_root_cutout * R, R, n_stations)
    dr = np.gradient(r)

    # State arrays
    a = np.full_like(r, 0.05, dtype=float)
    ap = np.full_like(r, 0.02, dtype=float)

    chord = np.asarray(chord, dtype=float)
    twist_deg = np.asarray(twist_deg, dtype=float)
    if chord.shape != r.shape or twist_deg.shape != r.shape:
        raise ValueError("chord and twist_deg must be arrays of length n_stations matching the internal r grid.")

    # Iterate induction
    for _ in range(max_iter):
        a_old = a.copy()
        ap_old = ap.copy()

        Vt = omega * r
        vi = a * Vt
        Vax = V_inf + vi

        # (swirl not used in kinematics in your original code; kept as-is)
        phi = np.arctan2(Vax, np.maximum(Vt, 1e-9))
        F = prandtl_tip_loss(B, r, R, phi)

        W = np.sqrt(Vax**2 + Vt**2)
        alpha = twist_deg - np.degrees(phi)

        Re = rho * W * chord / np.maximum(mu, 1e-12)
        Ma = W / np.maximum(a_sound, 1e-12)

        CL = np.empty_like(r)
        CD = np.empty_like(r)
        for i in range(r.size):
            CL[i], CD[i] = get_CL_CD(airfoil, alpha[i], Re[i], Ma[i])

        q = 0.5 * rho * W**2
        Lp = q * chord * CL
        Dp = q * chord * CD

        dTdr = B * (Lp * np.cos(phi) - Dp * np.sin(phi))
        dQdr = B * r * (Lp * np.sin(phi) + Dp * np.cos(phi))

        # Momentum update (your original style)
        denom = 4.0 * np.pi * rho * r * np.abs(Vax) * F
        denom = np.maximum(denom, 1e-9)

        vi_new = dTdr / denom
        a_new = vi_new / np.maximum(Vt, 1e-9)
        a_new = np.clip(a_new, -0.8, 0.8)

        denom_q = 4.0 * np.pi * rho * (r**3) * np.abs(Vax) * F * np.maximum(omega, 1e-9)
        denom_q = np.maximum(denom_q, 1e-9)
        ap_new = dQdr / denom_q
        ap_new = np.clip(ap_new, -0.8, 0.8)

        a = (1.0 - relax) * a + relax * a_new
        ap = (1.0 - relax) * ap + relax * ap_new

        if max(np.max(np.abs(a - a_old)), np.max(np.abs(ap - ap_old))) < tol:
            break

    # Integrate totals
    T = np.sum(dTdr * dr)
    Q = np.sum(dQdr * dr)
    P = Q * omega

    out = {
        "r": r,
        "dr": dr,
        "chord": chord,
        "twist_deg": twist_deg,
        "phi": phi,
        "alpha": alpha,
        "W": W,
        "Re": Re,
        "Ma": Ma,
        "dTdr": dTdr,
        "dQdr": dQdr,
        "Vax": Vax,
        "T": np.array([float(T)]),
        "Q": np.array([float(Q)]),
        "P": np.array([float(P)]),
        "a": a,
        "ap": ap,
    }
    return out


def coaxial_bemt_fixed(
    *,
    rho: float,
    mu: float,
    a_sound: float,
    T_total_target: float,
    R: float,
    B: int,
    omega: float,
    chord_upper: np.ndarray | None = None,
    chord_lower: np.ndarray | None = None,
    chord: np.ndarray | None = None,  # backward compatible (shared chord)
    twist_upper_deg: np.ndarray = None,
    twist_lower_deg: np.ndarray = None,
    V_inf: float = 0.0,
    airfoil: int = 2412,
    spacing: float = 0.2,          # [m] rotor separation (reserved)
    wake_factor: float = 2.0,      # far-wake velocity factor heuristic
    n_stations: int = 30,
    r_root_cutout: float = 0.1,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Coaxial BEMT with one-way coupling:
    - Upper rotor solved in freestream V_inf.
    - Lower rotor sees increased axial inflow equal to:
        V_inf_lower = V_inf + wake_factor * mean(vi_upper)
      where vi_upper ~ (Vax_upper - V_inf).

    Thrust split:
      T_upper = T_lower = 0.5 * T_total_target
    """
    # Each rotor shares thrust equally by default
    T_each = 0.5 * float(T_total_target)

    # Backward-compatible chord handling:
    # - New API: provide chord_upper and chord_lower separately.
    # - Old API: provide chord (applied to both).
    if chord_upper is None or chord_lower is None:
        if chord is None:
            raise ValueError("Provide either chord (shared) or chord_upper and chord_lower (independent).")
        chord_upper = chord
        chord_lower = chord

    upper = bemt_single(
        rho=rho, mu=mu, a_sound=a_sound,
        T_target=T_each,
        R=R, B=B, omega=omega,
        chord=chord_upper,
        twist_deg=twist_upper_deg,
        V_inf=V_inf,
        airfoil=airfoil,
        n_stations=n_stations,
        r_root_cutout=r_root_cutout,
    )

    # Estimate induced increment and pass to lower as added inflow (uniform disk-average)
    vi_upper = upper["Vax"] - float(V_inf)
    V_inf_lower_eff = float(V_inf) + wake_factor * float(np.mean(vi_upper))

    lower = bemt_single(
        rho=rho, mu=mu, a_sound=a_sound,
        T_target=T_each,
        R=R, B=B, omega=omega,
        chord=chord_lower,
        twist_deg=twist_lower_deg,
        V_inf=V_inf_lower_eff,
        airfoil=airfoil,
        n_stations=n_stations,
        r_root_cutout=r_root_cutout,
    )

    T_tot = float(upper["T"][0] + lower["T"][0])
    P_tot = float(upper["P"][0] + lower["P"][0])

    return {
        "upper": upper,
        "lower": lower,
        "totals": {
            "T": np.array([T_tot]),
            "P": np.array([P_tot]),
        },
    }