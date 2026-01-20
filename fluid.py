from dataclasses import dataclass, field
import numpy as np

@dataclass
class Fluid:
    altitude: float = 0.0  # Altitude in meters
    rho: float = field(init=False)  # Air density [kg/m^3]
    a: float = field(init=False)    # Speed of sound [m/s]
    nu: float = field(init=False)   # Kinematic viscosity [m^2/s]

    def __post_init__(self):
        """Compute air properties using ISA (up to 11 km)."""

        # --- Constants ---
        T0 = 288.15       # Sea level standard temperature [K]
        P0 = 101325.0     # Sea level standard pressure [Pa]
        L  = 0.0065       # Temperature lapse rate [K/m]
        R  = 287.058      # Specific gas constant for dry air [J/(kg·K)]
        g  = 9.80665      # Gravity [m/s^2]
        mu0 = 1.7894e-5   # Dynamic viscosity at 288.15 K [kg/(m·s)]
        S = 110.4         # Sutherland's constant [K]

        h = self.altitude

        # --- Temperature and pressure ---
        if h <= 11000:  # Troposphere
            T = T0 - L * h
            P = P0 * (T / T0) ** (g / (R * L))
        else:  # Isothermal layer (simple extension)
            T = 216.65
            P = P0 * (T / T0) ** (g / (R * L)) * \
                np.exp(-g * (h - 11000) / (R * T))

        # --- Air density ---
        rho = P / (R * T)

        # --- Dynamic viscosity (Sutherland's formula) ---
        mu = mu0 * (T / 288.15) ** 1.5 * (288.15 + S) / (T + S)

        # --- Derived quantities ---
        a = (1.4 * R * T) ** 0.5     # Speed of sound [m/s]
        nu = mu / rho                # Kinematic viscosity [m^2/s]

        # --- Store results ---
        self.rho = rho
        self.a = a
        self.nu = nu

    def __repr__(self):
        return (f"Fluid(rho={self.rho:.3f}, a={self.a:.1f}, nu={self.nu:.2e}, "
                f"altitude={self.altitude})")
