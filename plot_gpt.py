import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

import propeller

def plot_blade_planform_normalized(yaml_path, title="Blade Planform (Normalized r/R)"):
    """
    Loads a Propeller from YAML and plots the blade planform in non-dimensional coordinates.
    X-axis: r/R (0 to 1)
    Y-axis: x/R (Chordwise position normalized by Radius)
    """

    # 1. Load optimized rotor
    if not os.path.exists(yaml_path):
        print(f"Error: File not found at {yaml_path}")
        return

    rot = propeller.Propeller.load_from_yaml(yaml_path)
    R = float(rot.radius)

    # 2. Get Coordinates
    # r_R is already normalized (0 to 1)
    r_R = np.asarray(rot.r_R, dtype=float)
    
    # Calculate dimensional chord [m] first: (chord_values * c_ref)
    c_dim = np.asarray(rot.chord, dtype=float) * float(rot.c_ref)
    
    # Normalize chord by Radius to get c/R
    c_R = c_dim / R

    # 3. Define LE/TE referenced to 3/4-chord line at x=0
    # x_LE = -0.75 * c
    # x_TE = +0.25 * c
    # We normalize these by R as well
    x_le_norm = -0.75 * c_R
    x_te_norm = +0.25 * c_R

    # 4. Spline for smooth plotting
    # Create a smooth vector from min(r/R) to max(r/R)
    r_R_fine = np.linspace(r_R.min(), r_R.max(), 400)
    
    le_spline = CubicSpline(r_R, x_le_norm, bc_type="natural")
    te_spline = CubicSpline(r_R, x_te_norm, bc_type="natural")
    
    x_le_fine = le_spline(r_R_fine)
    x_te_fine = te_spline(r_R_fine)

    # 5. Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot the spline edges
    ax.plot(r_R_fine, x_le_fine, 'b-', label="Leading Edge")
    ax.plot(r_R_fine, x_te_fine, 'b-', label="Trailing Edge")
    
    # Fill the blade area
    ax.fill_between(r_R_fine, x_le_fine, x_te_fine, color='cyan', alpha=0.3)

    # Add the 3/4 chord line (x=0)
    ax.axhline(0.0, color='k', linestyle="--", alpha=0.5, label="3/4-chord line")

    # Formatting
    ax.set_title(title)
    ax.set_xlabel("Non-dimensional Radius ($r/R$)")
    ax.set_ylabel("Non-dimensional Chordwise Position ($x/R$)")
    ax.legend(loc='upper right')
    ax.grid(True)
    
    # Ensure aspect ratio is equal so the blade shape isn't distorted
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Point this to your optimized output file
    # Example: "optimized_prop_upper.yaml" or "data/optimized_prop_upper.yaml"
    filename = "pybemt_optimized_30stations.yaml" 
    
    # Check if file exists in current dir, otherwise try data/
    if not os.path.exists(filename):
        filename = os.path.join("data", filename)
        
    plot_blade_planform_normalized(filename, title="Optimized Blade Planform (Upper Rotor)")