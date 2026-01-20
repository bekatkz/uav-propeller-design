import math
import os
import subprocess
import tempfile

import aerosandbox as asb
import neuralfoil as nf
import numpy as np


def get_CL_CD_from_neuralfoil(airfoil: int, alpha: float, Re: float, Ma: float):
    """
    Uses NeuralFoil to get CL and CD for a given airfoil, angle of attack, Reynolds number, and Mach number.
    
    Parameters
    ----------
    airfoil : int
        Airfoil name (e.g., '4412').
    alpha : float
        Angle of attack in degrees.
    Re : float
        Reynolds number
    Ma : float
        Freestream Mach number.

    Returns
    -------
    CL, CD : tuple of floats
        Lift and drag coefficients.
    """
    airfoil = str("naca"+str(airfoil))
    af = asb.Airfoil(airfoil)

    aero = af.get_aero_from_neuralfoil(alpha=alpha, mach=Ma, model_size="large", Re=Re)
    CL = aero['CL']
    CD = aero['CD']

    return float(CL), float(CD)



def get_CL_CD_from_dat(filepath, alpha):
    """
    Reads an AeroDyn/QBlade .dat airfoil file and returns CL and CD for a given angle of attack.

    Parameters
    ----------
    filepath : str
        Path to the .dat file.
    alpha : float
        Angle of attack (degrees).

    Returns
    -------
    CL : float
        Lift coefficient corresponding to the given alpha.
    CD : float
        Drag coefficient corresponding to the given alpha.
    """

    AoA = []
    CL = []
    CD = []
    data_started = False

    with open(filepath, 'r') as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue

            parts = stripped.split()
            if len(parts) == 3:
                try:
                    a, cl, cd = map(float, parts)
                    data_started = True
                    AoA.append(a)
                    CL.append(cl)
                    CD.append(cd)
                except ValueError:
                    # Skip non-numeric lines
                    continue
            elif data_started:
                # Stop if we’ve left the numeric section
                break

    if not AoA:
        raise ValueError("No aerodynamic data found in the .dat file.")

    # Find nearest AoA
    AoA = np.array(AoA)
    CL = np.array(CL)
    CD = np.array(CD)

    idx = (np.abs(AoA - alpha)).argmin()

    return float(CL[idx]), float(CD[idx])


def access_xfoil_data(airfoil: int, alpha: float, Ma: float):
    """
    Reads the most recent XFOIL polar output (polar.out),
    extracts CL and CD for the given alpha,
    and clears the file afterward.

    Parameters
    ----------
    airfoil : int
        Name of the airfoil (e.g., "4412").
    alpha : float
        Angle of attack in degrees.

    Returns
    -------
    CL, CD : tuple of floats
        Lift and drag coefficients from polar.out
    """
    polar_file = "polar.out"
    if not os.path.exists(polar_file):
        raise FileNotFoundError(f"{polar_file} not found. Run XFOIL first.")

    CL, CD = None, None

    with open(polar_file, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        for ln in reversed(lines):  # look from bottom (most recent)
            parts = ln.split()
            if len(parts) >= 3:
                try:
                    alpha_val = float(parts[0])
                    cl_val = float(parts[1])
                    cd_val = float(parts[2])
                    # Take the line that matches or is closest to the requested alpha
                    if abs(alpha_val - alpha) < 0.25:  # tolerance in degrees
                        CL, CD = cl_val, cd_val
                        break
                except ValueError:
                    continue

    if CL is None or CD is None:
        ma_local_r = min(round(Ma * 20.0) / 20.0, 0.95)
        CL = get_CL(airfoil, alpha, ma_local_r)
        CD = get_CD(airfoil, alpha, ma_local_r)
    # # Clear the file for next use
    # open(polar_file, "w").close()

    return CL, CD



def get_CL(airfoil, alpha, MA):
    directory = f"NACA_{airfoil}"
    filename = f"NACA_{airfoil}_MA={MA}"
    file_path = os.path.join(directory, filename)

    with open(file_path, 'r') as file:
        # Skip headers
        for _ in range(6):
            next(file)

        alphas, CLs = [], []
        for line in file:
            columns = line.split()
            if len(columns) > 1:
                try:
                    alphas.append(float(columns[0]))
                    CLs.append(float(columns[1]))
                except ValueError:
                    continue

    if not alphas:
        raise ValueError(f"No valid CL data found in {file_path}")

    # If the exact alpha exists, return directly
    if alpha in alphas:
        return CLs[alphas.index(alpha)]

    # Find floor and ceil indices
    alpha_floor = math.floor(alpha)
    alpha_ceil = math.ceil(alpha)

    # Clamp to data bounds
    if alpha_floor < min(alphas):
        return CLs[0]
    if alpha_ceil > max(alphas):
        return CLs[-1]

    # Find corresponding CL values
    CL_floor = CLs[alphas.index(alpha_floor)]
    CL_ceil  = CLs[alphas.index(alpha_ceil)]

    # Linear interpolation
    CL_interp = CL_floor + (CL_ceil - CL_floor) * ((alpha - alpha_floor) / (alpha_ceil - alpha_floor))
    return CL_interp


def get_CD(airfoil, alpha, MA):
    directory = f"NACA_{airfoil}"
    filename = f"NACA_{airfoil}_MA={MA}"
    file_path = os.path.join(directory, filename)
    
    with open(file_path, 'r') as file:
        # Skip headers
        for _ in range(6):
            next(file)

        alphas, CDs = [], []
        for line in file:
            columns = line.split()
            if len(columns) > 2:
                try:
                    alphas.append(float(columns[0]))
                    CDs.append(float(columns[2]))
                except ValueError:
                    continue

    if not alphas:
        raise ValueError(f"No valid CD data found in {file_path}")

    if alpha in alphas:
        return CDs[alphas.index(alpha)]

    alpha_floor = math.floor(alpha)
    alpha_ceil = math.ceil(alpha)

    if alpha_floor < min(alphas):
        return CDs[0]
    if alpha_ceil > max(alphas):
        return CDs[-1]

    CD_floor = CDs[alphas.index(alpha_floor)]
    CD_ceil  = CDs[alphas.index(alpha_ceil)]

    CD_interp = CD_floor + (CD_ceil - CD_floor) * ((alpha - alpha_floor) / (alpha_ceil - alpha_floor))
    return CD_interp


def get_CM(alpha, MA):
    file_path = f'NACA_4412/NACA_4412_MA={MA}'
    
    with open(file_path, 'r') as file:
        # Skipping headers
        for _ in range(6):
            next(file)

        # Reading data and extracting CL values
        for line in file:
            columns = line.split()

            if len(columns) > 1:
                alpha_val = columns[0]

                # Check if alpha_val can be converted to float
                try:
                    alpha_val_float = float(alpha_val)
                except ValueError:
                    continue  # Skip this line if alpha_val is not a valid float

                if alpha_val_float == alpha:
                    return float(columns[4])
    
    return None




def run_xfoil(airfoil: str, alpha: float, Re: float, Mach: float = 0.0, xfoil_path: str = "xfoil"):
    """
    Runs XFOIL for a specified airfoil, angle of attack, Reynolds number, and Mach number,
    but does not parse or return any results.

    Parameters
    ----------
    airfoil_name : int
        Airfoil name (e.g., '4412'). Function will look for './NACA_4412/NACA_4412.dat'.
    alpha : float
        Angle of attack in degrees.
    Re : float
        Reynolds number (typically chord-based).
    Mach : float, optional
        Freestream Mach number (default: 0.0).
    xfoil_path : str, optional
        Path to the XFOIL executable (default assumes it's on PATH).
    """
    # Construct airfoil file path assuming structure: <airfoil>/<airfoil>.dat
    directory = f"NACA_{airfoil}"
    filename = f"NACA_{airfoil}.dat"
    airfoil_file = os.path.join(directory, filename)

    if not os.path.exists(airfoil_file):
        raise FileNotFoundError(f"Airfoil file not found: {airfoil_file}")

   
    polar_file =  "polar.out"

    # Prepare XFOIL input script (no indentation!)
    script = f"""LOAD {airfoil_file}
PANE
OPER
VISC {Re}
MACH {Mach}
NCRIT 9
ITER 200
PACC
{polar_file}
y

ALFA {alpha}
PACC

QUIT
"""


    # Write script to temp file
    script_file = "xfoil.in"
    with open(script_file, "w") as f:
        f.write(script)

    # Run XFOIL silently (no return parsing)
    with open(os.devnull, "w") as devnull:
        subprocess.run([xfoil_path], input=script.encode('utf-8'),
                        stdout=devnull, stderr=devnull, check=False)

    CL, CD = access_xfoil_data(airfoil, alpha=alpha, Ma=Mach)

    return CL, CD


def main():
    #CL, CD = run_xfoil("NACA_4412", alpha=5.0, Re=1e6, Mach=0.1)
    CL, CD = get_CL_CD_from_neuralfoil(4412, alpha=5, Re=1e5, Ma=0.1)
    
    print(f"CL: {CL}, CD: {CD}")

if __name__ == "__main__":
    main()