import numpy as np
from functools import lru_cache

_AIRFOIL_CACHE = {}

# bin sizes (tune)
DALPHA = 0.25     # deg
DLOGRE = 0.10     # log10(Re)
DMA = 0.02        # Mach

def get_CL_CD_from_neuralfoil(airfoil, alpha, Re, Ma):
    import aerosandbox as asb
    airfoil = int(airfoil)
    if airfoil not in _AIRFOIL_CACHE:
        _AIRFOIL_CACHE[airfoil] = asb.Airfoil(f"naca{airfoil}")
    af = _AIRFOIL_CACHE[airfoil]
    aero = af.get_aero_from_neuralfoil(alpha=float(alpha), Re=float(Re), mach=float(Ma), model_size="medium")
    return float(aero["CL"]), float(aero["CD"])

@lru_cache(maxsize=50000)
def _cached_base(airfoil, alpha_q, logRe_q, Ma_q):
    Re_q = 10.0 ** float(logRe_q)
    return get_CL_CD_from_neuralfoil(airfoil, alpha_q, Re_q, Ma_q)

def get_CL_CD(airfoil, alpha, Re, Ma):
    # quantize Re and Ma for cache keys
    logRe_q = round(np.round(np.log10(max(Re, 1.0)) / DLOGRE) * DLOGRE, 6)
    Ma_q    = round(np.round(max(Ma, 0.0) / DMA) * DMA, 6)

    # quantize alpha to two neighbors
    a0 = np.floor(alpha / DALPHA) * DALPHA
    a1 = a0 + DALPHA
    a0 = round(float(a0), 6)
    a1 = round(float(a1), 6)

    cl0, cd0 = _cached_base(int(airfoil), a0, logRe_q, Ma_q)
    cl1, cd1 = _cached_base(int(airfoil), a1, logRe_q, Ma_q)

    # interpolate in alpha
    w = 0.0 if abs(a1 - a0) < 1e-12 else (alpha - a0) / (a1 - a0)
    w = 0.0 if w < 0.0 else (1.0 if w > 1.0 else w)

    CL = (1 - w) * cl0 + w * cl1
    CD = (1 - w) * cd0 + w * cd1
    return CL, CD
