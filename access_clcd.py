import numpy as np
from functools import lru_cache
import aerosandbox as asb
import neuralfoil as nf

_AIRFOIL_CACHE = {}

# bin sizes (tune) - INCREASED FOR SPEED
DALPHA = 0.50     # deg
DLOGRE = 0.25     # log10(Re)
DMA = 0.05        # Mach

def get_CL_CD_from_neuralfoil(airfoil, alpha, Re, Ma):
    airfoil = int(airfoil)
    if airfoil not in _AIRFOIL_CACHE:
        _AIRFOIL_CACHE[airfoil] = asb.Airfoil(f"naca{airfoil}")
    af = _AIRFOIL_CACHE[airfoil]
    # Changed to xsmall for massive speed boost
    aero = af.get_aero_from_neuralfoil(alpha=float(alpha), Re=float(Re), mach=float(Ma), model_size="xsmall")
    return float(aero["CL"]), float(aero["CD"])

# Changed maxsize to None so it never forgets a calculation
@lru_cache(maxsize=None)
def _cached_base(airfoil, alpha_q, logRe_q, Ma_q):
    Re_q = 10.0 ** float(logRe_q)
    return get_CL_CD_from_neuralfoil(airfoil, alpha_q, Re_q, Ma_q)

def get_CL_CD(airfoil, alpha, Re, Ma):
    # Keep your original get_CL_CD function exactly as it was here for compatibility
    ALPHA_MIN, ALPHA_MAX = -10.0, 12.0
    MACH_MIN,  MACH_MAX  = 0.0, 0.75
    RE_MIN,    RE_MAX    = 1.0, 5e6

    alpha = float(np.clip(alpha, ALPHA_MIN, ALPHA_MAX))
    Ma    = float(np.clip(Ma, MACH_MIN,  MACH_MAX))
    Re    = float(np.clip(Re, RE_MIN,    RE_MAX))

    logRe_q = round(np.round(np.log10(Re) / DLOGRE) * DLOGRE, 6)
    Ma_q    = round(np.round(Ma / DMA) * DMA, 6)

    a0 = np.floor(alpha / DALPHA) * DALPHA
    a1 = a0 + DALPHA

    a0 = round(float(np.clip(a0, ALPHA_MIN, ALPHA_MAX)), 6)
    a1 = round(float(np.clip(a1, ALPHA_MIN, ALPHA_MAX)), 6)

    cl0, cd0 = _cached_base(int(airfoil), a0, logRe_q, Ma_q)

    if abs(a1 - a0) < 1e-12:
        return cl0, cd0

    cl1, cd1 = _cached_base(int(airfoil), a1, logRe_q, Ma_q)

    w = (alpha - a0) / (a1 - a0)
    w = 0.0 if w < 0.0 else (1.0 if w > 1.0 else w)

    CL = (1.0 - w) * cl0 + w * cl1
    CD = (1.0 - w) * cd0 + w * cd1
    return CL, CD

def get_CL_CD_array(airfoil: int, alpha_deg: np.ndarray, Re: np.ndarray, Ma: np.ndarray):
    """Vectorized wrapper to avoid Python loop overhead"""
    airfoil = int(airfoil)

    ALPHA_MIN, ALPHA_MAX = -10.0, 12.0
    MACH_MIN,  MACH_MAX  = 0.0, 0.75
    RE_MIN,    RE_MAX    = 1.0, 5e6

    alpha = np.clip(alpha_deg.astype(float), ALPHA_MIN, ALPHA_MAX)
    Ma    = np.clip(Ma.astype(float),        MACH_MIN,  MACH_MAX)
    Re    = np.clip(Re.astype(float),        RE_MIN,    RE_MAX)

    logRe_q = np.round(np.log10(Re) / DLOGRE) * DLOGRE
    Ma_q    = np.round(Ma / DMA) * DMA

    a0 = np.floor(alpha / DALPHA) * DALPHA
    a1 = np.clip(a0 + DALPHA, ALPHA_MIN, ALPHA_MAX)

    w = (alpha - a0) / np.maximum(a1 - a0, 1e-12)
    w = np.clip(w, 0.0, 1.0)

    a0 = np.round(a0, 6); a1 = np.round(a1, 6)
    logRe_q = np.round(logRe_q, 6)
    Ma_q    = np.round(Ma_q, 6)

    keys0 = list(zip(a0.tolist(), logRe_q.tolist(), Ma_q.tolist()))
    keys1 = list(zip(a1.tolist(), logRe_q.tolist(), Ma_q.tolist()))

    lut = {}
    def eval_key(k):
        if k not in lut:
            alpha_q, logRe_q_i, Ma_q_i = k
            lut[k] = _cached_base(airfoil, float(alpha_q), float(logRe_q_i), float(Ma_q_i))
        return lut[k]

    cl0 = np.empty_like(alpha, dtype=float)
    cd0 = np.empty_like(alpha, dtype=float)
    cl1 = np.empty_like(alpha, dtype=float)
    cd1 = np.empty_like(alpha, dtype=float)

    for i, k in enumerate(keys0):
        cl0[i], cd0[i] = eval_key(k)
    for i, k in enumerate(keys1):
        cl1[i], cd1[i] = eval_key(k)

    CL = (1.0 - w) * cl0 + w * cl1
    CD = (1.0 - w) * cd0 + w * cd1
    return CL, CD