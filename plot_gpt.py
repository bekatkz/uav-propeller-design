import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import propeller

def plot_blade_true_1to1(yaml_path):
    """
    Plots the blade planform with a strict 1:1 data aspect ratio.
    Manually sets c_ref = 0.15 to correct the scale.
    """
    # 1. Load Data
    if not os.path.exists(yaml_path):
        print(f"Error: YAML file not found at {yaml_path}")
        return

    rot = propeller.Propeller.load_from_yaml(yaml_path)
    
    # --- THE FIX: FORCE THE REFERENCE CHORD ---
    rot.c_ref = 0.15  # <--- CRITICAL FIX: Matches your optimization config
    # ------------------------------------------

    # Get Physical dimensions
    R_tip = float(rot.radius)
    r_coords = np.array(rot.r_R) * R_tip
    chord_len = np.array(rot.chord) * rot.c_ref  # Now this will be ~0.375m max

    # 2. Define Geometry relative to Pitch Axis (c/4)
    y_le =  0.25 * chord_len
    y_te = -0.75 * chord_len

    # 3. Smoothing
    r_fine = np.linspace(r_coords.min(), r_coords.max(), 500)
    spline_le = CubicSpline(r_coords, y_le, bc_type='natural')
    spline_te = CubicSpline(r_coords, y_te, bc_type='natural')
    y_le_fine = spline_le(r_fine)
    y_te_fine = spline_te(r_fine)

    # 4. Create Plot
    fig, ax = plt.subplots(figsize=(15, 6))

    # Plot Edges
    ax.plot(r_fine, y_le_fine, 'b-', linewidth=2, label='Leading Edge')
    ax.plot(r_fine, y_te_fine, 'b-', linewidth=2, label='Trailing Edge')
    
    # Fill Blade
    ax.fill_between(r_fine, y_te_fine, y_le_fine, color='cyan', alpha=0.3, label='Blade Surface')
    
    # Pitch Axis
    ax.axhline(0, color='k', linestyle='--', alpha=0.6, label='Pitch Axis (c/4)')

    # 5. STRICT EQUAL ASPECT RATIO
    ax.set_aspect('equal', adjustable='box')

    # Formatting
    ax.set_title(f"TRUE 1:1 Physical Scale (c_ref fixed to 0.15m)\nRadius: {R_tip:.3f} m | Max Chord: {np.max(chord_len):.3f} m", fontsize=14)
    ax.set_xlabel("Radius [m]", fontsize=12)
    ax.set_ylabel("Chordwise Width [m]", fontsize=12)
    
    # Force ticks to be consistent
    ax.grid(True, which='major', color='gray', linestyle='-', linewidth=0.8)
    ax.minorticks_on()
    ax.grid(True, which='minor', color='gray', linestyle=':', linewidth=0.5)

    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Update this path to your saved lower rotor file
    filename = "data/pybemt_optimized_ehang_polished_lower.yaml"
    plot_blade_true_1to1(filename)