# -*- coding: utf-8 -*-
"""
=============================================================================
AERODYNAMIC LOOKUP TABLE (LUT) GENERATOR
=============================================================================
Description:
    This module provides the Lift (CL) and Drag (CD) coefficients for a given 
    airfoil at a specific Angle of Attack, Reynolds number, and Mach number.

Theory & Design Choices:
    1. NeuralFoil Integration: Uses MIT's AeroSandbox/NeuralFoil to get highly 
       accurate viscous aerodynamic data.
    2. Dynamic Caching (LUT): Calling a neural network directly inside the BEMT 
       integration loop (which runs millions of times) is computationally impossible. 
       Instead, this script quantizes the continuous inputs into discrete "bins", 
       evaluates the neural network only once per bin, and caches the result.
    3. Linear Interpolation: When BEMT requests a continuous Angle of Attack, 
       this script fetches the two nearest cached bins and linearly interpolates 
       between them to provide a smooth, continuous output without the NN penalty.
=============================================================================
"""

import numpy as np
from functools import lru_cache
import aerosandbox as asb
import neuralfoil as nf

# =============================================================================
# 1. CONFIGURATION & TUNING PARAMETERS
# =============================================================================

# Bin sizes for the dynamic Lookup Table (LUT). 
# Increasing these speeds up the code but reduces interpolation accuracy.
DALPHA = 0.50     # Angle of Attack bin size [deg]
DLOGRE = 0.25     # Reynolds number log-scale bin size [log10(Re)]
DMA    = 0.05     # Mach number bin size

# Physical limits for the aerodynamic database
ALPHA_MIN, ALPHA_MAX = -10.0, 12.0
MACH_MIN,  MACH_MAX  = 0.0, 0.75
RE_MIN,    RE_MAX    = 1.0, 5e6

# Global cache to prevent re-initializing the Airfoil object
_AIRFOIL_CACHE = {}


# =============================================================================
# 2. CORE NEURALFOIL ENGINE & CACHE
# =============================================================================

def get_CL_CD_from_neuralfoil(airfoil: int, alpha: float, Re: float, Ma: float) -> tuple[float, float]:
    """
    Directly evaluates the NeuralFoil neural network for the requested state.
    Uses the 'xsmall' model to massively reduce matrix multiplication overhead.
    """
    airfoil = int(airfoil)
    
    # Cache the airfoil object to save initialization time
    if airfoil not in _AIRFOIL_CACHE:
        _AIRFOIL_CACHE[airfoil] = asb.Airfoil(f"naca{airfoil}")
    af = _AIRFOIL_CACHE[airfoil]
    
    # Evaluate the neural network
    aero = af.get_aero_from_neuralfoil(
        alpha=float(alpha), 
        Re=float(Re), 
        mach=float(Ma), 
        model_size="xsmall"
    )
    return float(aero["CL"]), float(aero["CD"])

# Use lru_cache(maxsize=None) so the computer never forgets a previously evaluated bin.
# This makes the BEMT solver faster with every iteration.
@lru_cache(maxsize=None)
def _cached_base(airfoil: int, alpha_q: float, logRe_q: float, Ma_q: float) -> tuple[float, float]:
    """
    The cached wrapper. It takes the quantized (binned) inputs, converts logRe 
    back to absolute Re, and calls the neural network.
    """
    Re_q = 10.0 ** float(logRe_q)
    return get_CL_CD_from_neuralfoil(airfoil, alpha_q, Re_q, Ma_q)


# =============================================================================
# 3. SCALAR INTERFACE (SINGLE BLADE ELEMENT)
# =============================================================================

def get_CL_CD(airfoil: int, alpha: float, Re: float, Ma: float) -> tuple[float, float]:
    """
    Retrieves CL and CD for a single blade element by quantizing the inputs,
    looking up the nearest cached bounds, and linearly interpolating the result.
    """
    # 1. Clip inputs to prevent the neural network from extrapolating wildly
    alpha = float(np.clip(alpha, ALPHA_MIN, ALPHA_MAX))
    Ma    = float(np.clip(Ma, MACH_MIN,  MACH_MAX))
    Re    = float(np.clip(Re, RE_MIN,    RE_MAX))

    # 2. Quantize Re and Mach to the nearest bin
    logRe_q = round(np.round(np.log10(Re) / DLOGRE) * DLOGRE, 6)
    Ma_q    = round(np.round(Ma / DMA) * DMA, 6)

    # 3. Find the bounding Angle of Attack bins (a0 is below, a1 is above)
    a0 = np.floor(alpha / DALPHA) * DALPHA
    a1 = a0 + DALPHA
    a0 = round(float(np.clip(a0, ALPHA_MIN, ALPHA_MAX)), 6)
    a1 = round(float(np.clip(a1, ALPHA_MIN, ALPHA_MAX)), 6)

    # 4. Fetch the aerodynamic data for the lower bound
    cl0, cd0 = _cached_base(int(airfoil), a0, logRe_q, Ma_q)

    # If the requested alpha falls exactly on a bin edge, return immediately
    if abs(a1 - a0) < 1e-12:
        return cl0, cd0

    # 5. Fetch the aerodynamic data for the upper bound
    cl1, cd1 = _cached_base(int(airfoil), a1, logRe_q, Ma_q)

    # 6. Calculate the interpolation weight (w) and interpolate
    w = (alpha - a0) / (a1 - a0)
    w = 0.0 if w < 0.0 else (1.0 if w > 1.0 else w)

    CL = (1.0 - w) * cl0 + w * cl1
    CD = (1.0 - w) * cd0 + w * cd1
    
    return CL, CD


# =============================================================================
# 4. VECTORIZED INTERFACE (FULL BLADE SPAN)
# =============================================================================

def get_CL_CD_array(airfoil: int, alpha_deg: np.ndarray, Re: np.ndarray, Ma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized wrapper for get_CL_CD. 
    Theory: Processes the entire blade (all radial stations) simultaneously 
    using numpy arrays to drastically minimize slow Python loops.
    """
    airfoil = int(airfoil)

    # 1. Clip inputs array-wise
    alpha = np.clip(alpha_deg.astype(float), ALPHA_MIN, ALPHA_MAX)
    Ma    = np.clip(Ma.astype(float),        MACH_MIN,  MACH_MAX)
    Re    = np.clip(Re.astype(float),        RE_MIN,    RE_MAX)

    # 2. Quantize arrays
    logRe_q = np.round(np.log10(Re) / DLOGRE) * DLOGRE
    Ma_q    = np.round(Ma / DMA) * DMA

    # 3. Establish bin bounds
    a0 = np.floor(alpha / DALPHA) * DALPHA
    a1 = np.clip(a0 + DALPHA, ALPHA_MIN, ALPHA_MAX)

    # 4. Calculate interpolation weights array
    w = (alpha - a0) / np.maximum(a1 - a0, 1e-12)
    w = np.clip(w, 0.0, 1.0)

    # Prevent float precision errors from missing the cache
    a0 = np.round(a0, 6)
    a1 = np.round(a1, 6)
    logRe_q = np.round(logRe_q, 6)
    Ma_q    = np.round(Ma_q, 6)

    # 5. Create unique keys for the cache lookup
    keys0 = list(zip(a0.tolist(), logRe_q.tolist(), Ma_q.tolist()))
    keys1 = list(zip(a1.tolist(), logRe_q.tolist(), Ma_q.tolist()))

    # Local temporary cache for this specific blade evaluation
    lut = {}
    def eval_key(k):
        if k not in lut:
            alpha_q, logRe_q_i, Ma_q_i = k
            lut[k] = _cached_base(airfoil, float(alpha_q), float(logRe_q_i), float(Ma_q_i))
        return lut[k]

    # Pre-allocate output arrays
    cl0 = np.empty_like(alpha, dtype=float)
    cd0 = np.empty_like(alpha, dtype=float)
    cl1 = np.empty_like(alpha, dtype=float)
    cd1 = np.empty_like(alpha, dtype=float)

    # 6. Evaluate keys (this triggers the neural net ONLY for new bins)
    for i, k in enumerate(keys0):
        cl0[i], cd0[i] = eval_key(k)
    for i, k in enumerate(keys1):
        cl1[i], cd1[i] = eval_key(k)

    # 7. Apply vectorized interpolation
    CL = (1.0 - w) * cl0 + w * cl1
    CD = (1.0 - w) * cd0 + w * cd1
    
    return CL, CD