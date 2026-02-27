import numpy as np
from functools import lru_cache

_AIRFOIL_CACHE = {}

# bin sizes (tune)
DALPHA = 0.5     # deg
DLOGRE = 0.20     # log10(Re)
DMA = 0.05        # Mach

def get_CL_CD_from_neuralfoil(airfoil, alpha, Re, Ma):
    import aerosandbox as asb
    airfoil = int(airfoil)
    if airfoil not in _AIRFOIL_CACHE:
        _AIRFOIL_CACHE[airfoil] = asb.Airfoil(f"naca{airfoil}")
    af = _AIRFOIL_CACHE[airfoil]
    aero = af.get_aero_from_neuralfoil(alpha=float(alpha), Re=float(Re), mach=float(Ma), model_size="small")
    return float(aero["CL"]), float(aero["CD"])

@lru_cache(maxsize=200000)
def _cached_base(airfoil, alpha_q, logRe_q, Ma_q):
    Re_q = 10.0 ** float(logRe_q)
    return get_CL_CD_from_neuralfoil(airfoil, alpha_q, Re_q, Ma_q)

def get_CL_CD(airfoil, alpha, Re, Ma):
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

    # clamp endpoints too (prevents querying beyond limits at alpha=ALPHA_MAX)
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