from dataclasses import dataclass, field
from typing import List
import yaml
import numpy as np


@dataclass
class Propeller:
    nblades: int
    solidity: float
    radius: float  # [m]
    omega: float   # [rad/s]
    r_R: List[float] = field(default_factory=list)  # r/R radial stations
    chord: List[float] = field(default_factory=list)  # c/c_ref chord distribution
    pitch: List[float] = field(default_factory=list)  # pitch distribution
    sweep: List[float] = field(default_factory=list)  # sweep distribution
    airfoil: List[int] = field(default_factory=list)  # airfoil index list (4-digit NACA code)
    dr: List[float] = field(init=False)               # computed automatically
    c_ref: float = field(default=None)                # computed from solidity if not given

    def __post_init__(self):
        # --- Compute reference chord if not provided ---
        if self.c_ref is None:
            self.c_ref = (self.solidity * np.pi * self.radius) / self.nblades

        # If the user gives dimensional radii, normalize to r/R
        if len(self.r_R) == 0:
            raise ValueError("r/R list cannot be empty.")
        # --- Compute dr automatically ---
        r = np.array(self.r_R, dtype=float)
        dr = np.diff(r) * self.radius
        # repeat the last interval so dr has same length as r
        dr = np.append(dr, dr[-1])
        self.dr = dr.tolist()


    @classmethod
    def load_from_yaml(cls, filepath: str):
        """Load propeller data from a YAML file, ignoring dr."""
        with open(filepath, "r") as file:
            data = yaml.safe_load(file)
        prop_data = data["propeller"]
        geom = prop_data["geometry"]

        return cls(
            nblades=prop_data["nblades"],
            solidity=prop_data["solidity"],
            radius=np.array(prop_data["radius"]),
            omega=prop_data["omega"]*np.pi/30,
            r_R=np.array(geom["radii"]),
            pitch=np.array(geom["pitch"]),
            chord=np.array(geom["chord"]),
            sweep=np.array(geom["sweep"]),
            airfoil=np.array(geom["airfoil"]),
        )
    def save_to_yaml(self, filepath: str):
        """Save propeller data to a YAML file."""
        data = {
            "propeller": {
                "nblades": int(self.nblades),
                "solidity": float(self.solidity),
                "radius": float(self.radius),
                "omega": float(self.omega*30/np.pi),
                "geometry": {
                    "radii": to_float_list(self.r_R),
                    "pitch": to_float_list(self.pitch),
                    "chord": to_float_list(self.chord),
                    "sweep": to_float_list(self.sweep),
                    "airfoil": to_float_list(self.airfoil),
                }
            }
        }
        with open(filepath, "w") as file:
            yaml.dump(data, file)



def to_float_list(x):
    """Convert a NumPy array or list to a list of floats."""
    if isinstance(x, np.ndarray):
        return x.astype(float).tolist()
    elif isinstance(x, list):
        return [float(i) for i in x]
    else:
        return [float(x)]  # handle a single scalar


def main():
    test_prop = Propeller.load_from_yaml("data/pybemt_tmotor28_modified.yaml")
    # print("r/R:", test_prop.r_R)
    # print("dr:", test_prop.dr)
    # print("Reference chord c_ref:", test_prop.c_ref)
    # print(test_prop.chord[-1] * test_prop.c_ref )
    #print(np.array(test_prop.r_R)*test_prop.radius)
    #print(np.array(test_prop.chord)*test_prop.c_ref)
    print("radius = ", np.array(test_prop.r_R) * test_prop.radius)
    print("chord:", test_prop.c_ref*np.array(test_prop.chord))

    test_prop.chord = (np.array(test_prop.chord)*1.1).tolist()
    test_prop.save_to_yaml("data/pybemt_tmotor28_modified.yaml")
if __name__ == "__main__":
    main()
