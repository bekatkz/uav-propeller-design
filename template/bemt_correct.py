import matplotlib.pyplot as plt
import numpy as np

import fluid
import propeller
from access_clcd import get_CL_CD_from_neuralfoil


def bemt(propeller, fluid, V_inf, omega):
    ## parameters
    radius = propeller.radius
    chord = np.array(propeller.chord, dtype=float) * propeller.c_ref
    r_R = np.array(propeller.r_R, dtype=float)
    beta = propeller.pitch
    nblades = propeller.nblades
    rho = fluid.rho
    airfoil = propeller.airfoil
    r = r_R * radius
    # initializations
    
    R_cutout = 0.15 * radius    
    
    tol = 1e-5
    max_iter = 500
    # Compute flow angle:
    dT_dr = np.zeros(len(r_R))
    dM_dr = np.zeros(len(r_R))
    print(f"Starting BEMT: Omega={omega:.1f} rad/s, V_inf={V_inf}")
    v_i = 5.0
    relax = 0.1
    a_prime = 0.01
    for i in range(len(r_R)):
        if r[i] < R_cutout:
            dT_dr[i] = 0
            dM_dr[i] = 0
            continue
        
        sigmal = nblades * chord[i] / (2 * np.pi * r[i])
                
        if r[i] - radius == 0:
            dT_dr[i] = 0
            dM_dr[i] = 0
            continue

        for iter in range(max_iter):           
            
            V_axial = v_i + V_inf
            V_tangential = omega * r[i] * (1 - a_prime)

            phi = np.arctan2(V_axial, V_tangential)

            U = np.sqrt(V_axial**2 + V_tangential**2)
            
            # compute local angle of attack
            alpha = beta[i] - np.degrees(phi)
            
            Re = U * chord[i] / fluid.nu
            Ma = U / fluid.a

            # get Cl, Cd from neuralfoil
            cl, cd = get_CL_CD_from_neuralfoil(airfoil[i], alpha=alpha, Re=Re, Ma=Ma)
            
            C_n = cl * np.cos(phi) - cd * np.sin(phi)
            
            C_t = cl * np.sin(phi) + cd * np.cos(phi)

            # Prandtl's tip loss factor
            f = (nblades / 2) * (radius - r[i]) / (r[i] * np.sin(phi))
            f = max(f, 1e-4)
        
            F = min(2 / np.pi * np.arccos(np.exp(-f)), 1.0)
            # Calculate induction factor a_prime to compute the torque
            kappap = 4 * F * np.sin(phi) * np.cos(phi) / (sigmal * C_t)
            a_prime_new = 1 / (kappap + 1)
   
            a_prime = relax * a_prime_new + (1 - relax) * a_prime
            
            if F*(V_inf**2*F+C_n*sigmal*U**2) < 0:
                v_i_new = np.sign(C_n) * 1/(2*F) * (np.sqrt(abs(F*(V_inf**2*F+C_n*sigmal*U**2)))-V_inf*F)
            else:
                v_i_new = np.sign(C_n) * 1/(2*F) * (np.sqrt(F*(V_inf**2*F+C_n*sigmal*U**2))-V_inf*F)
            

            if abs(v_i_new - v_i) < tol:
                break
            if iter==max_iter-1:
                print("max_iter reached - not converged")
            v_i = relax * v_i_new + (1 - relax) * v_i

        # BET
        dT_dr[i] = 0.5 * rho * U**2 * chord[i] * nblades * C_n 
        dM_dr[i] = 0.5 * rho * U**2 * chord[i] * nblades * C_t * r[i]
        
        if F*(V_inf**2*F+C_n*sigmal*U**2) < 0:
            print("Negative sectional lift detected, taking absolute of root at r/R="+str(r_R[i]) )
        
    # Integrate loads   
    Thrust = np.trapezoid(dT_dr, x=r_R * radius)
    Torque = np.trapezoid(dM_dr, x=r_R * radius)
    return Thrust, Torque


def main():
    test_propeller = propeller.Propeller.load_from_yaml("data/pybemt_tmotor28.yaml")
    test_fluid = fluid.Fluid(0)

    OMEGA = 5000 * 2 * np.pi / 60

    Thrust, Torque = bemt(test_propeller, test_fluid, V_inf=5, omega=OMEGA)
    print(f"Thrust: {Thrust:.2f} N, Torque: {Torque:.4f} Nm")

if __name__ == "__main__":
    main()
