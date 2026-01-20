import math
import os
import subprocess
import aerosandbox as asb
import neuralfoil as nf
import numpy as np

# --- 1. THE CACHE (Prevents re-loading 50,000 times) ---
_AIRFOIL_CACHE = {}

def get_CL_CD_from_neuralfoil(airfoil: int, alpha: float, Re: float, Ma: float):
    """
    Optimized version with Caching.
    """
    airfoil_code = int(airfoil)
    
    # Check if we already loaded this airfoil
    if airfoil_code not in _AIRFOIL_CACHE:
        # If not, load it and save it to the cache
        name = f"naca{airfoil_code}"
        _AIRFOIL_CACHE[airfoil_code] = asb.Airfoil(name)
    
    # Retrieve from cache (Instant!)
    af = _AIRFOIL_CACHE[airfoil_code]

    # Run Analysis
    aero = af.get_aero_from_neuralfoil(alpha=alpha, mach=Ma, model_size="medium", Re=Re)
    
    CL = aero['CL']
    CD = aero['CD']

    return float(CL), float(CD)

# (Keep your other functions like get_CL_CD_from_dat below this if needed)