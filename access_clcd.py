import math
import os
import subprocess
import aerosandbox as asb
import neuralfoil as nf
import numpy as np

# --- CACHE ---
# This dictionary prevents us from reloading the airfoil 50,000 times
_AIRFOIL_CACHE = {}

def get_CL_CD_from_neuralfoil(airfoil: int, alpha: float, Re: float, Ma: float):
    """
    Optimized version with Caching.
    """
    airfoil_code = int(airfoil)
    
    # 1. Check if we already loaded this airfoil
    if airfoil_code not in _AIRFOIL_CACHE:
        # If not, load it and save it to the cache
        name = f"naca{airfoil_code}"
        _AIRFOIL_CACHE[airfoil_code] = asb.Airfoil(name)
    
    # 2. Retrieve from cache
    af = _AIRFOIL_CACHE[airfoil_code]

    # 3. Run Analysis
    # 'model_size="medium"' is faster than "large" and accurate enough for optimization
    aero = af.get_aero_from_neuralfoil(alpha=alpha, mach=Ma, model_size="medium", Re=Re)
    
    CL = aero['CL']
    CD = aero['CD']

    # Ensure we return floats (NeuralFoil sometimes returns 1-element arrays)
    return float(CL), float(CD)

# (Keep the rest of your functions like get_CL_CD_from_dat if you want, 
# but the one above is the only one used by the optimizer)