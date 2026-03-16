# -*- coding: utf-8 -*-
"""
=============================================================================
POST-PROCESSING & PLOTTING UTILITY
=============================================================================
Description:
    Standalone plotting utility for the coaxial optimization project.
    Reads the compressed .npz physics output and generates professional, 
    publication-ready aerodynamic charts.

Usage:
    python plotting.py <path/to/results.npz> [optional_out_dir]
=============================================================================
"""

from __future__ import annotations
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Global Color Standard for Consistency
COLOR_UPPER = 'tab:blue'
COLOR_LOWER = 'tab:orange'


def _savefig(path: str):
    """Helper function to save plots cleanly and free memory."""
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches='tight')
    plt.close()


def plot_blade_planform_34chord(d: dict, out_dir: str, filename: str = "fig_blade_planform.png"):
    """
    Creates a horizontal blade planform plot with two stacked subplots.
    Upper blade is on top, Lower blade is on bottom. Both use true 1:1 physical scale.
    """
    r_R = d["r_R"].astype(float)
    R = float(d["R"])
    r = r_R * R

    if "chord_U_m" in d.files and "chord_L_m" in d.files:
        cU = d["chord_U_m"].astype(float)
        cL = d["chord_L_m"].astype(float)
        
        fig, (axU, axL) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # --- TOP PLOT: UPPER BLADE ---
        axU.axhline(0.0, linestyle=":", color="red", label="3/4-chord line")
        axU.plot(r, -0.75 * cU, color=COLOR_UPPER, linewidth=1.5)
        axU.plot(r, +0.25 * cU, color=COLOR_UPPER, linewidth=1.5)
        axU.fill_between(r, -0.75 * cU, +0.25 * cU, color=COLOR_UPPER, alpha=0.3, label="Upper Blade")
        axU.set_aspect('equal', adjustable='box')
        axU.grid(True, linestyle="--", alpha=0.6)
        axU.set_ylabel("x (chordwise) [m]")
        axU.set_title("Upper Blade Planform (True 1:1 Scale)")
        axU.legend(loc="upper right")
        
        # --- BOTTOM PLOT: LOWER BLADE ---
        axL.axhline(0.0, linestyle=":", color="red", label="3/4-chord line")
        axL.plot(r, -0.75 * cL, color=COLOR_LOWER, linewidth=1.5)
        axL.plot(r, +0.25 * cL, color=COLOR_LOWER, linewidth=1.5)
        axL.fill_between(r, -0.75 * cL, +0.25 * cL, color=COLOR_LOWER, alpha=0.3, label="Lower Blade")
        axL.set_aspect('equal', adjustable='box')
        axL.grid(True, linestyle="--", alpha=0.6)
        axL.set_xlabel("r (spanwise) [m]")
        axL.set_ylabel("x (chordwise) [m]")
        axL.set_title("Lower Blade Planform (True 1:1 Scale)")
        axL.legend(loc="upper right")
        
    else:
        # Fallback for old shared-chord files
        fig, ax = plt.subplots(figsize=(10, 4))
        c = d["chord_m"].astype(float) if "chord_m" in d.files else d["c_over_R"] * R
        ax.axhline(0.0, linestyle=":", color="red", label="3/4-chord line")
        ax.plot(r, -0.75 * c, color="black", linewidth=1.5)
        ax.plot(r, +0.25 * c, color="black", linewidth=1.5)
        ax.fill_between(r, -0.75 * c, +0.25 * c, color="gray", alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_xlabel("r (spanwise) [m]")
        ax.set_ylabel("x (chordwise) [m]")
        ax.set_title("Blade Planform (True 1:1 Scale)")

    _savefig(os.path.join(out_dir, filename))


def make_required_plots(npz_path: str, out_dir: str):
    """Reads the NPZ archive and triggers all plotting functions."""
    d = np.load(npz_path, allow_pickle=False)
    os.makedirs(out_dir, exist_ok=True)
    r_R = d["r_R"]

    # ==========================================
    # 1. GEOMETRY PLOTS
    # ==========================================
    
    # Planform (True Scale)
    plot_blade_planform_34chord(d, out_dir, "fig_blade_planform.png")

    # Chord Distribution
    plt.figure(figsize=(7.2, 4.6))
    if "c_over_R_U" in d.files and "c_over_R_L" in d.files:
        plt.plot(r_R, d["c_over_R_U"], label="Upper Chord (c/R)", color=COLOR_UPPER, linewidth=2)
        plt.plot(r_R, d["c_over_R_L"], label="Lower Chord (c/R)", color=COLOR_LOWER, linewidth=2)
        plt.title("Chord Distribution (Decoupled)")
    else:
        cR = d["c_over_R"] if "c_over_R" in d.files else d["chord_m"]/d["R"]
        plt.plot(r_R, cR, label="Shared Chord (c/R)", color='green', linewidth=2)
        plt.title("Chord Distribution (Shared)")
    plt.grid(True)
    plt.xlabel("r/R [-]")
    plt.ylabel("c/R [-]")
    plt.legend()
    _savefig(os.path.join(out_dir, "fig_chord_distribution.png"))

    # Twist Distribution
    if "twistU_deg" in d.files:
        plt.figure(figsize=(7.2, 4.6))
        plt.plot(r_R, d["twistU_deg"], label="Upper β", color=COLOR_UPPER, linewidth=2)
        plt.plot(r_R, d["twistL_deg"], label="Lower β", color=COLOR_LOWER, linewidth=2)
        plt.grid(True)
        plt.xlabel("r/R [-]")
        plt.ylabel("Twist / pitch β [deg]")
        plt.title("Twist distribution (upper & lower)")
        plt.legend()
        _savefig(os.path.join(out_dir, "fig_twist_distribution.png"))

    # ==========================================
    # 2. AERODYNAMIC PLOTS
    # ==========================================

    # Angle of Attack
    if "alphaU_deg" in d.files:
        plt.figure(figsize=(7.2, 4.6))
        plt.plot(r_R, d["alphaU_deg"], label="Upper α", color=COLOR_UPPER, linewidth=2)
        plt.plot(r_R, d["alphaL_deg"], label="Lower α", color=COLOR_LOWER, linewidth=2)
        plt.grid(True)
        plt.xlabel("r/R [-]")
        plt.ylabel("Angle of attack α [deg]")
        plt.title("Angle of attack distribution")
        plt.legend()
        _savefig(os.path.join(out_dir, "fig_angle_of_attack.png"))

    # Inflow Angle
    if "phiU_deg" in d.files:
        plt.figure(figsize=(7.2, 4.6))
        plt.plot(r_R, d["phiU_deg"], label="Upper φ", color=COLOR_UPPER, linewidth=2)
        plt.plot(r_R, d["phiL_deg"], label="Lower φ", color=COLOR_LOWER, linewidth=2)
        plt.grid(True)
        plt.xlabel("r/R [-]")
        plt.ylabel("Inflow angle φ [deg]")
        plt.title("Inflow angle distribution")
        plt.legend()
        _savefig(os.path.join(out_dir, "fig_inflow_angle_phi.png"))

    # Axial Velocity (Wake Interference)
    if "VaxU" in d.files:
        plt.figure(figsize=(7.2, 4.6))
        plt.plot(r_R, d["VaxU"], label="Upper V_ax (Clean Air)", color=COLOR_UPPER, linewidth=2)
        plt.plot(r_R, d["VaxL"], label="Lower V_ax (Wake Air)", color=COLOR_LOWER, linewidth=2)
        plt.grid(True)
        plt.xlabel("r/R [-]")
        plt.ylabel("Axial Velocity V_ax [m/s]")
        plt.title("Axial Velocity / Wake Interference")
        plt.legend()
        _savefig(os.path.join(out_dir, "fig_axial_velocity.png"))

    # Lift Coefficient
    if "CL_upper" in d.files:
        plt.figure(figsize=(7.2, 4.6))
        plt.plot(r_R, d["CL_upper"], label="Upper C_L", color=COLOR_UPPER, linewidth=2)
        plt.plot(r_R, d["CL_lower"], label="Lower C_L", color=COLOR_LOWER, linewidth=2)
        plt.axhline(1.2, color='red', linestyle=':', label="Typical Stall Limit (~1.2)")
        plt.grid(True)
        plt.xlabel("r/R [-]")
        plt.ylabel("Lift Coefficient C_L [-]")
        plt.title("Spanwise Lift Coefficient (Stall Margin)")
        plt.legend()
        _savefig(os.path.join(out_dir, "fig_lift_coefficient.png"))

    # Drag Coefficient
    if "CD_upper" in d.files:
        plt.figure(figsize=(7.2, 4.6))
        plt.plot(r_R, d["CD_upper"], label="Upper C_D", color=COLOR_UPPER, linewidth=2)
        plt.plot(r_R, d["CD_lower"], label="Lower C_D", color=COLOR_LOWER, linewidth=2)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.xlabel("r/R [-]")
        plt.ylabel("Drag Coefficient C_D [-]")
        plt.title("Spanwise Drag Coefficient (Profile Drag)")
        plt.legend()
        _savefig(os.path.join(out_dir, "fig_drag_coefficient.png")) 

    # Lift-to-Drag Ratio (L/D)
    if "CL_upper" in d.files and "CD_upper" in d.files:
        L_D_upper = d["CL_upper"] / d["CD_upper"]
        L_D_lower = d["CL_lower"] / d["CD_lower"]
        plt.figure(figsize=(7.2, 4.6))
        plt.plot(r_R, L_D_upper, label="Upper L/D", color=COLOR_UPPER, linewidth=2)
        plt.plot(r_R, L_D_lower, label="Lower L/D", color=COLOR_LOWER, linewidth=2)
        plt.grid(True)
        plt.xlabel("r/R [-]")
        plt.ylabel("Lift-to-Drag Ratio (CL/CD)")
        plt.title("Sectional Aerodynamic Efficiency (L/D)")
        plt.legend()
        _savefig(os.path.join(out_dir, "fig_lift_to_drag_ratio.png"))

    # ==========================================
    # 3. PERFORMANCE & LOADING PLOTS
    # ==========================================

    # Radial Thrust Loading
    if "dTdrU" in d.files:
        plt.figure(figsize=(7.2, 4.6))
        plt.plot(r_R, d["dTdrU"], label="Upper dT/dr", color=COLOR_UPPER, linewidth=2)
        plt.plot(r_R, d["dTdrL"], label="Lower dT/dr", color=COLOR_LOWER, linewidth=2)
        plt.grid(True)
        plt.xlabel("r/R [-]")
        plt.ylabel("dT/dr [N/m]")
        plt.title("Radial Thrust Loading")
        plt.legend()
        _savefig(os.path.join(out_dir, "fig_thrust_loading_dTdr.png"))

    # Torque Loading
    if "dQdrU" in d.files:
        plt.figure(figsize=(7.2, 4.6))
        plt.plot(r_R, d["dQdrU"], label="Upper dQ/dr", color=COLOR_UPPER, linewidth=2)
        plt.plot(r_R, d["dQdrL"], label="Lower dQ/dr", color=COLOR_LOWER, linewidth=2)
        plt.grid(True)
        plt.xlabel("r/R [-]")
        plt.ylabel("Torque Loading dQ/dr [N·m/m]")
        plt.title("Radial Torque Distribution")
        plt.legend()
        _savefig(os.path.join(out_dir, "fig_torque_loading_dQdr.png"))

    # ==========================================
    # 4. SWEEP PLOTS (Climb Performance)
    # ==========================================
    
    if "Vs" in d.files:
        Vs = d["Vs"]

        # Power vs Climb Speed
        if "P_sweep" in d.files and "P_excess_sweep" in d.files:
            plt.figure(figsize=(7.2, 4.6))
            plt.plot(Vs, d["P_sweep"], label="Total Battery Power Required", color="purple", linewidth=2)
            plt.plot(Vs, d["P_excess_sweep"], label="Aerodynamic Power (Spinning blades only)", color="green", linestyle="--", linewidth=2)
            plt.grid(True)
            plt.xlabel("Vertical Speed Vc [m/s] (Negative = Descent, Positive = Climb)")
            plt.ylabel("Power [W]")
            plt.title("Total Power vs. Aerodynamic Power")
            plt.legend()
            _savefig(os.path.join(out_dir, "fig_power_vs_climb_speed.png"))
            
            # Propulsive Efficiency (Climb only)
            if "T_sweep" in d.files:
                pos_mask = Vs > 0
                if np.any(pos_mask):
                    Vs_pos = Vs[pos_mask]
                    T_pos = d["T_sweep"][pos_mask]
                    P_pos = d["P_sweep"][pos_mask]
                    eta = (T_pos * Vs_pos) / P_pos
                    
                    plt.figure(figsize=(7.2, 4.6))
                    plt.plot(Vs_pos, eta * 100, label="Propulsive Efficiency η", color="purple", linewidth=2)
                    plt.grid(True, linestyle="--", alpha=0.7)
                    plt.xlabel("Climb speed Vc [m/s]")
                    plt.ylabel("Propulsive Efficiency [%]")
                    plt.title("Propulsive Efficiency vs. Climb Speed")
                    plt.legend()
                    _savefig(os.path.join(out_dir, "fig_propulsive_efficiency.png"))

        # RPM vs Speed
        if "RPM_sweep" in d.files:
            plt.figure(figsize=(7.2, 4.6))
            plt.plot(Vs, d["RPM_sweep"], color="black", linewidth=2)
            plt.grid(True)
            plt.xlabel("Climb speed Vc [m/s]")
            plt.ylabel("Trimmed RPM [-]")
            plt.title("Trimmed RPM vs climb speed")
            _savefig(os.path.join(out_dir, "fig_rpm_vs_climb_speed.png"))

        # Tip Mach vs Speed
        if "Mtip_sweep" in d.files:
            plt.figure(figsize=(7.2, 4.6))
            plt.plot(Vs, d["Mtip_sweep"], color="red", linewidth=2)
            plt.grid(True)
            plt.xlabel("Climb speed Vc [m/s]")
            plt.ylabel("Tip Mach [-]")
            plt.title("Tip Mach vs climb speed (trimmed)")
            _savefig(os.path.join(out_dir, "fig_tip_mach_vs_climb_speed.png"))


def _cli():
    if len(sys.argv) < 2:
        print("Usage: python plotting.py <path/to/results.npz> [out_dir]")
        raise SystemExit(2)
    npz_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) >= 3 else os.path.dirname(npz_path)
    make_required_plots(npz_path, out_dir)
    print(f"[OK] Wrote figures to: {out_dir}")

if __name__ == "__main__":
    _cli()