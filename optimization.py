import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
import copy
import os

# Import your project modules
import fluid
import propeller
import bemt_coaxial

# ==========================================
# 1. GEOMETRY GENERATORS (SMOOTHING)
# ==========================================

def get_high_res_radial_grid(n_points=50, r_root_norm=0.15):
    """Generates a smooth, high-resolution radial grid (r/R)."""
    return np.linspace(r_root_norm, 1.0, n_points)

def generate_linear_twist(r_R, beta_root, beta_tip):
    """
    Generates a linear twist distribution on the given r_R grid.
    """
    # Map min(r_R) -> beta_root, max(r_R) -> beta_tip
    r_min = r_R[0]
    r_max = r_R[-1]
    slope = (beta_tip - beta_root) / (r_max - r_min)
    twist = beta_root + slope * (r_R - r_min)
    return twist

def generate_spline_chord(r_R, chord_params, radius):
    """
    Generates chord distribution using a cubic spline.
    chord_params: [c_root, c_mid1, c_mid2, c_tip] (normalized c/R)
    """
    # 1. Define control points at 4 fixed stations
    r_stations = np.linspace(r_R[0], r_R[-1], 4)
    
    # 2. Create the spline function
    cs = CubicSpline(r_stations, chord_params, bc_type='natural')
    
    # 3. Evaluate on the actual (high-res) r_R grid
    c_R_dist = cs(r_R)
    
    # Safety: Ensure no negative chord, min 1mm
    c_R_dist = np.maximum(c_R_dist, 0.005) 
    
    return c_R_dist # Returns c/R

# ==========================================
# 2. OPTIMIZATION COST FUNCTION
# ==========================================

def objective_function(x, base_prop1, base_prop2, fluid_model, aircraft_data):
    """
    Variables x:
    [0:4] -> Chord params (c/R) for BOTH rotors
    [4:6] -> Rotor 1 Twist [root, tip]
    [6:8] -> Rotor 2 Twist [root, tip]
    [8]   -> RPM
    """
    # Unpack variables
    chord_params = x[0:4]
    tw1_params   = x[4:6]
    tw2_params   = x[6:8]
    rpm          = x[8]
    
    omega = rpm * 2 * np.pi / 60.0
    
    # --- HIGH-RES SMOOTHING ---
    # We ignore the original 10 points and create 40 new ones
    r_fine = get_high_res_radial_grid(n_points=40, r_root_norm=0.15)
    
    # Create temp props for analysis
    p1 = copy.deepcopy(base_prop1)
    p2 = copy.deepcopy(base_prop2)
    
    # Update Prop 1 Geometry
    p1.r_R = r_fine
    p1.omega = omega
    p1.pitch = generate_linear_twist(r_fine, tw1_params[0], tw1_params[1])
    # Note: Optimization uses physical chord (meters), bypassing c_ref
    p1.c_ref = 1.0 
    p1.chord = generate_spline_chord(r_fine, chord_params, p1.radius) * p1.radius
    
    # Update Prop 2 Geometry (Same chord, different twist)
    p2.r_R = r_fine
    p2.omega = omega
    p2.pitch = generate_linear_twist(r_fine, tw2_params[0], tw2_params[1])
    p2.c_ref = 1.0
    p2.chord = generate_spline_chord(r_fine, chord_params, p2.radius) * p2.radius
    
    # Interpolate airfoils (nearest neighbor for simplicity on fine grid)
    # Or just fill with the main airfoil code
    main_af = base_prop1.airfoil[1] # Take mid-span airfoil
    p1.airfoil = [main_af] * len(r_fine)
    p2.airfoil = [main_af] * len(r_fine)

    # --- RUN SOLVER ---
    try:
        totals, _, _, _ = bemt_coaxial.coaxial_bemt_fixed(
            p1, p2, fluid_model, 
            V_inf=aircraft_data['v_climb'], 
            omega1=omega, omega2=omega, 
            trim_rotor2=True, alpha_target_deg=3.0
        )
        
        # --- CALCULATE COST ---
        P_total = totals['Q_total'] * omega
        T_total = totals['T_total']
        T_req = aircraft_data['T_required_per_arm']
        
        # Soft Constraint: Penalty for missing thrust
        # If T < T_req, add huge penalty. If T > T_req, add small penalty (don't overpower too much)
        if T_total < T_req:
            penalty = 1000.0 * (T_req - T_total)**2
        else:
            penalty = 0.0
            
        return P_total + penalty

    except Exception:
        return 1e9 # Return high cost if solver crashes

# ==========================================
# 3. HELPER: FIND RPM FOR EXACT THRUST
# ==========================================
def trim_rpm_for_thrust(p1, p2, fl, V_inf, T_target):
    """Used for the final sweep to find exact power at different speeds."""
    rpm_min, rpm_max = 1000, 12000
    best_rpm = rpm_min
    
    for _ in range(15): # Bisection search
        rpm_mid = 0.5 * (rpm_min + rpm_max)
        om = rpm_mid * 2 * np.pi / 60
        p1.omega = om; p2.omega = om
        
        try:
            totals, _, _, _ = bemt_coaxial.coaxial_bemt_fixed(
                p1, p2, fl, V_inf, om, om, trim_rotor2=True
            )
            T_curr = totals['T_total']
        except:
            T_curr = 0
            
        if T_curr < T_target:
            rpm_min = rpm_mid
        else:
            rpm_max = rpm_mid
            best_rpm = rpm_mid
            
    return best_rpm, totals['Q_total'] * (best_rpm * 2 * np.pi / 60)

# ==========================================
# 4. MAIN OPTIMIZATION LOOP
# ==========================================
def main():
    print("--- STARTING OPTIMIZATION (With Smoothing) ---")
    
    # 1. Aircraft Inputs
    aircraft = {
        'mass': 5.0,        # kg
        'n_arms': 4,        # Quadcopter (4 coaxial pairs)
        'v_climb': 5.0,     # m/s
        'radius': 0.20,     # m
        'altitude': 0.0
    }
    aircraft['T_required_per_arm'] = (aircraft['mass'] * 9.81) / aircraft['n_arms']
    print(f"Target Thrust: {aircraft['T_required_per_arm']:.2f} N per arm")

    # 2. Setup Base Objects
    base_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(base_dir, "data", "pybemt_tmotor28.yaml")
    
    base_prop = propeller.Propeller.load_from_yaml(yaml_path)
    base_prop.radius = aircraft['radius']
    fl = fluid.Fluid(aircraft['altitude'])

    # 3. Define Bounds
    # Chord (c/R) [root, m1, m2, tip]
    # Twist (deg) [root, tip]
    bounds = [
        (0.05, 0.25), (0.05, 0.25), (0.05, 0.25), (0.02, 0.15), # Chord
        (10.0, 40.0), (0.0, 15.0),                              # Twist R1
        (10.0, 40.0), (0.0, 15.0),                              # Twist R2
        (2000, 8000)                                            # RPM
    ]
    
    # Initial Guess
    x0 = [0.1, 0.12, 0.1, 0.05,  20, 5,  20, 5,  4500]

    # 4. Run Optimization
    print("Optimizing... (This takes 30-60 seconds)")
    res = minimize(
        objective_function, x0, 
        args=(base_prop, base_prop, fl, aircraft),
        method='SLSQP', bounds=bounds,
        options={'maxiter': 40, 'disp': True}
    )
    
    print(f"\nOptimization Success: {res.success}")
    x_opt = res.x

    # ==========================================
    # 5. POST-PROCESS & PLOT
    # ==========================================
    # Reconstruct the optimized, SMOOTH propellers
    r_fine = get_high_res_radial_grid(50)
    omega_opt = x_opt[8] * 2 * np.pi / 60
    
    p1_opt = copy.deepcopy(base_prop); p1_opt.r_R = r_fine
    p2_opt = copy.deepcopy(base_prop); p2_opt.r_R = r_fine
    
    # Apply optimized geometry
    chord_dist = generate_spline_chord(r_fine, x_opt[0:4], p1_opt.radius)
    
    p1_opt.c_ref = 1.0; p1_opt.chord = chord_dist * p1_opt.radius
    p1_opt.pitch = generate_linear_twist(r_fine, x_opt[4], x_opt[5])
    p1_opt.airfoil = [base_prop.airfoil[1]] * 50
    p1_opt.omega = omega_opt

    p2_opt.c_ref = 1.0; p2_opt.chord = chord_dist * p2_opt.radius
    p2_opt.pitch = generate_linear_twist(r_fine, x_opt[6], x_opt[7])
    p2_opt.airfoil = [base_prop.airfoil[1]] * 50
    p2_opt.omega = omega_opt

    # Run Final Analysis
    totals, out1, out2, coupling = bemt_coaxial.coaxial_bemt_fixed(
        p1_opt, p2_opt, fl, aircraft['v_climb'], omega_opt, omega_opt,
        trim_rotor2=True, alpha_target_deg=3.0
    )

    # --- PLOTS ---
    # 1. Geometry
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(r_fine, p1_opt.chord * 1000, 'b-', linewidth=2)
    plt.title("Optimized Chord Distribution")
    plt.ylabel("Chord [mm]"); plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(r_fine, p1_opt.pitch, 'b-', label="Rotor 1")
    plt.plot(r_fine, p2_opt.pitch, 'r-', label="Rotor 2")
    plt.title("Optimized Twist Distribution")
    plt.ylabel("Twist [deg]"); plt.grid(True); plt.legend()
    plt.show()
    
    # 2. BEMT Results (Smooth!)
    bemt_coaxial.plot_diagnostics(out1, out2, totals, coupling, title_suffix=" (Optimized)")
    
    # 3. Power Sweep (Requirement)
    print("\nRunning Climb Speed Sweep...")
    speeds = np.arange(-5, 11, 2)
    powers = []
    for v in speeds:
        _, p_req = trim_rpm_for_thrust(p1_opt, p2_opt, fl, v, aircraft['T_required_per_arm'])
        powers.append(p_req)
        
    plt.figure()
    plt.plot(speeds, powers, 'o-')
    plt.xlabel("Climb Speed [m/s]"); plt.ylabel("Power [W]")
    plt.title("Power vs Climb Speed (Const Thrust)")
    plt.grid(True); plt.show()

if __name__ == "__main__":
    main()