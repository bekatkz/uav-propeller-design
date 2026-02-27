# -*- coding: utf-8 -*-
"""
plotting_coaxial.py

Standalone plotting utility for the coaxial optimization project.

Usage
-----
1) From optimization script:
    import plotting_coaxial as pc
    pc.make_required_plots(npz_path, out_dir)

2) Or run directly:
    python plotting_coaxial.py results/run_YYYYMMDD_HHMMSS/results.npz

This module ONLY reads the NPZ and writes figures. It does not run optimization.
"""

from __future__ import annotations
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def _savefig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def make_required_plots(npz_path: str, out_dir: str):
    """
    Generates the required plots:
      - chord distribution (c/R)
      - twist distribution (upper & lower)
      - inflow angle phi (upper & lower)
      - thrust loading dT/dr (upper & lower)
      - power vs climb speed (trimmed)
      - power excluding climb power (P - T*Vc)

    Also generates recommended plots if the NPZ contains the keys:
      - alpha distribution (upper & lower)
      - trimmed RPM vs climb speed
      - tip Mach vs climb speed
      
    """
    

    
    d = np.load(npz_path, allow_pickle=False)

    r_R = d["r_R"]

    # design-point geometry (UPDATED FOR INDEPENDENT CHORDS)
    # Assuming your npz file saves them as 'c_over_R_U' and 'c_over_R_L'
    # design-point geometry
# Support BOTH independent and shared chord storage

    if "c_over_R_U" in d.files and "c_over_R_L" in d.files:
        cR_U = d["c_over_R_U"]
        cR_L = d["c_over_R_L"]
    else:
        # fallback: shared chord distribution
        cR = d["c_over_R"]
        cR_U = cR
        cR_L = cR
    
    betaU = d["twistU_deg"]
    betaL = d["twistL_deg"]
    # design-point distributions
    phiU = d["phiU_deg"]
    phiL = d["phiL_deg"]
    dTdrU = d["dTdrU"]
    dTdrL = d["dTdrL"]

    # sweep
    Vs = d["Vs"]
    P_shaft = d["P_sweep"]
    P_excess = d["P_excess_sweep"]

    os.makedirs(out_dir, exist_ok=True)
    
    plot_blade_planform_34chord(npz_path, out_dir, filename="fig_blade_planform.png")

    # 1) Chord distribution (UPDATED PLOT)
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(r_R, cR_U, label="Upper Blade (c/R)", color='blue')
    plt.plot(r_R, cR_L, label="Lower Blade (c/R)", color='orange', linestyle='--')
    plt.grid(True)
    plt.xlabel("r/R [-]")
    plt.ylabel("c/R [-]")
    plt.title("Chord distribution (Upper vs Lower)")
    plt.legend()
    _savefig(os.path.join(out_dir, "fig_chord_distribution.png"))

    # 2) Twist distribution
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(r_R, betaU, label="Upper β")
    plt.plot(r_R, betaL, label="Lower β")
    plt.grid(True)
    plt.xlabel("r/R [-]")
    plt.ylabel("Twist / pitch β [deg]")
    plt.title("Twist distribution (upper & lower)")
    plt.legend()
    _savefig(os.path.join(out_dir, "fig_twist_distribution.png"))

    # 3) Inflow angle distribution
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(r_R, phiU, label="Upper φ")
    plt.plot(r_R, phiL, label="Lower φ")
    plt.grid(True)
    plt.xlabel("r/R [-]")
    plt.ylabel("Inflow angle φ [deg]")
    plt.title("Inflow angle distribution")
    plt.legend()
    _savefig(os.path.join(out_dir, "fig_inflow_angle_phi.png"))

    # 4) Radial thrust loading
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(r_R, dTdrU, label="Upper dT/dr")
    plt.plot(r_R, dTdrL, label="Lower dT/dr")
    plt.grid(True)
    plt.xlabel("r/R [-]")
    plt.ylabel("dT/dr [N/m]")
    plt.title("Radial thrust loading")
    plt.legend()
    _savefig(os.path.join(out_dir, "fig_thrust_loading_dTdr.png"))

    # 5) Power vs climb speed
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(Vs, P_shaft)
    plt.grid(True)
    plt.xlabel("Climb speed Vc [m/s]")
    plt.ylabel("Shaft power P [W]")
    plt.title("Power vs climb speed (trimmed RPM)")
    _savefig(os.path.join(out_dir, "fig_power_vs_climb_speed.png"))

    # 6) Power excluding climb power
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(Vs, P_excess)
    plt.grid(True)
    plt.xlabel("Climb speed Vc [m/s]")
    plt.ylabel("P_shaft - T*Vc [W]")
    plt.title("Power excluding climb power (trimmed RPM)")
    _savefig(os.path.join(out_dir, "fig_power_excluding_climb.png"))

    # Recommended: alpha
    if "alphaU_deg" in d.files and "alphaL_deg" in d.files:
        plt.figure(figsize=(7.2, 4.6))
        plt.plot(r_R, d["alphaU_deg"], label="Upper α")
        plt.plot(r_R, d["alphaL_deg"], label="Lower α")
        plt.grid(True)
        plt.xlabel("r/R [-]")
        plt.ylabel("Angle of attack α [deg]")
        plt.title("Angle of attack distribution")
        plt.legend()
        _savefig(os.path.join(out_dir, "fig_angle_of_attack.png"))

    # Recommended: RPM vs speed
    if "RPM_sweep" in d.files:
        plt.figure(figsize=(7.2, 4.6))
        plt.plot(Vs, d["RPM_sweep"])
        plt.grid(True)
        plt.xlabel("Climb speed Vc [m/s]")
        plt.ylabel("Trimmed RPM [-]")
        plt.title("Trimmed RPM vs climb speed")
        _savefig(os.path.join(out_dir, "fig_rpm_vs_climb_speed.png"))

    # Recommended: Mtip vs speed
    if "Mtip_sweep" in d.files:
        plt.figure(figsize=(7.2, 4.6))
        plt.plot(Vs, d["Mtip_sweep"])
        plt.grid(True)
        plt.xlabel("Climb speed Vc [m/s]")
        plt.ylabel("Tip Mach [-]")
        plt.title("Tip Mach vs climb speed (trimmed)")
        _savefig(os.path.join(out_dir, "fig_tip_mach_vs_climb_speed.png"))


def _cli():
    if len(sys.argv) < 2:
        print("Usage: python plotting_coaxial.py <path/to/results.npz> [out_dir]")
        raise SystemExit(2)
    npz_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) >= 3 else os.path.dirname(npz_path)
    make_required_plots(npz_path, out_dir)
    print(f"[OK] Wrote figures to: {out_dir}")


def plot_blade_planform_34chord(npz_path: str, out_dir: str, filename: str = "fig_blade_planform.png"):
    """
    Creates a blade planform (LE/TE outline) plot with the 3/4-chord line fixed at x=0
    for BOTH the upper and lower blades.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    d = np.load(npz_path, allow_pickle=False)

    r_R = d["r_R"].astype(float)
    R = float(d["R"])
    
    # Read independent chords in meters
    # Adjust the keys here to match what you save in your npz file!
    # Read chord distribution (support both independent and shared)

    if "chordU_m" in d.files and "chordL_m" in d.files:
        cU = d["chordU_m"].astype(float)
        cL = d["chordL_m"].astype(float)
    else:
        c = d["chord_m"].astype(float)
        cU = c
        cL = c

    r = r_R * R  # spanwise coordinate [m]

    # Upper blade geometry
    x_le_U = -0.75 * cU
    x_te_U = +0.25 * cU
    
    # Lower blade geometry
    x_le_L = -0.75 * cL
    x_te_L = +0.25 * cL

    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(6.8, 5.2))
    
    # Plot Upper Blade
    plt.plot(x_le_U, r, label="Upper LE", color="blue")
    plt.plot(x_te_U, r, label="Upper TE", color="blue")
    plt.fill_betweenx(r, x_le_U, x_te_U, color="blue", alpha=0.15)
    
    # Plot Lower Blade
    plt.plot(x_le_L, r, label="Lower LE", color="orange", linestyle="--")
    plt.plot(x_te_L, r, label="Lower TE", color="orange", linestyle="--")
    plt.fill_betweenx(r, x_le_L, x_te_L, color="orange", alpha=0.15)
    
    plt.axvline(0.0, linestyle=":", color="black", label="3/4-chord line (x=0)")

    plt.grid(True)
    plt.xlabel("x (chordwise) [m]")
    plt.ylabel("r (spanwise) [m]")
    plt.title("Blade planform (Upper vs Lower)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=220)
    plt.close()


if __name__ == "__main__":
    _cli()
