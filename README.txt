# Coaxial UAV Rotor Optimisation using Blade Element Momentum Theory

## Project Overview

This project presents the aerodynamic design and optimisation of a coaxial rotor system for a 650 kg unmanned aerial vehicle (UAV) operating in hover and climb conditions.

The objective is to **minimise the required propulsion power while generating the necessary thrust** by optimising the blade geometry.

The rotor performance is evaluated using **Blade Element Momentum Theory (BEMT)** combined with a **coaxial wake interaction model**, and the blade geometry is optimised using **numerical optimisation algorithms**.

The implementation is written in **Python** and includes:

* aerodynamic simulation
* airfoil lookup modelling
* coaxial rotor interaction modelling
* numerical optimization
* automated post-processing and visualisation

---

# Aircraft Design Case

The rotor system is optimised for the following UAV configuration.

### Aircraft Parameters

| Parameter           | Value    |
| ------------------- | -------- |
| Aircraft mass       | 650 kg   |
| Coaxial rotor units | 8        |
| Total rotors        | 16       |
| Blades per rotor    | 2        |
| Disk loading        | 160 N/m² |

### Flight Condition

| Parameter   | Value     |
| ----------- | --------- |
| Altitude    | 500 m     |
| Climb speed | 3 m/s     |
| Airfoil     | NACA 2412 |

The rotor radius is determined from the **disk loading requirement and thrust demand**, rather than being predefined.

---

# Methodology

## Blade Element Momentum Theory (BEMT)

The aerodynamic performance of each rotor is evaluated using **Blade Element Momentum Theory**, which combines:

### Blade Element Theory

The blade is divided into **30 radial stations**.
For each section, the solver computes:

* local flow velocity
* inflow angle
* angle of attack
* lift and drag forces

### Momentum Theory

Momentum theory determines the **induced velocity through the rotor disk** required to produce the thrust.

The solver iteratively balances both models until convergence.

The implementation includes:

* Prandtl **tip loss correction**
* **root loss correction**
* **induced velocity iteration**
* **swirl velocity modelling**

---

## Coaxial Rotor Interaction

In a coaxial configuration, the **lower rotor operates in the wake of the upper rotor**.

The model assumes:

* upper rotor operates in **clean inflow**
* lower rotor sees **accelerated wake flow**

The wake velocity is estimated using a **far-wake approximation with a decay factor**.

This approach captures the dominant aerodynamic interaction while keeping the simulation computationally efficient.

---

# Airfoil Modelling

The rotor blades use the **NACA 2412 airfoil**.

Aerodynamic coefficients are obtained using **NeuralFoil**, a neural-network-based airfoil solver.

To improve computational efficiency:

* aerodynamic states are **cached in a lookup table**
* coefficients are **interpolated between bins**

This avoids expensive neural network evaluations within the BEMT loop.

---

# Optimisation Strategy

The blade geometry is optimised using a **two-stage optimisation process**.

### Stage 1 — Global Search

A **Differential Evolution** algorithm explores the design space and identifies the region of minimum power.

The objective function includes penalties for:

* thrust error
* stall conditions
* geometric irregularities

---

### Stage 2 — Local Refinement

The best design from Stage 1 is refined using **COBYLA-constrained optimisation**.

Constraints include:

* thrust matching
* blade loading limits
* stall margin
* torque balance between rotors

The final solution produces a **physically realistic rotor geometry** with minimal shaft power.

---

# Blade Geometry Parameterisation

The blade geometry is defined using spline control points.

### Chord Distribution

A **4-point cubic spline** defines the blade chord:

* small root chord
* maximum chord near **0.35R**
* smooth taper towards the tip

### Twist Distribution

A **5-point spline** defines blade pitch:

* high pitch near the root
* decreasing pitch towards the tip

This produces approximately a uniform angle of attack across the blade span, improving aerodynamic efficiency.

---

# Example Results

Example optimised performance:

| Metric                  | Value   |
| ----------------------- | ------- |
| Thrust per coaxial unit | 797 N   |
| Power per coaxial unit  | 11.5 kW |
| Total aircraft power    | 92 kW   |
| Propulsive efficiency   | ~20 %   |
| Tip Mach number         | 0.46    |

The optimised rotor operates within aerodynamic limits and avoids stall.

---

# Generated Plots

The code automatically generates aerodynamic and performance plots, including:

### Geometry

* blade platform
* chord distribution
* twist distribution

### Aerodynamics

* angle of attack distribution
* inflow angle
* lift coefficient
* drag coefficient
* lift-to-drag ratio

### Performance

* thrust loading along the blade span
* torque loading
* power vs climb speed
* propulsive efficiency

---

# Running the Project

### Install dependencies

```
pip install numpy scipy matplotlib aerosandbox neuralfoil pyyaml
```

The script will:

1. Perform rotor optimisation
2. Export the optimised geometry
3. generate aerodynamic plots

Outputs are saved in:

```
results/run_timestamp/
```

---

# Engineering Skills Demonstrated

This project demonstrates experience in:

* rotorcraft aerodynamics
* Blade Element Momentum Theory
* coaxial rotor modelling
* aerodynamic optimization
* numerical methods
* Python scientific computing
* data visualization

---

# Author

Bexultan Tokkozha
Aerospace Engineering Student
Technical University of Munich (TUM)

Focus areas:

* UAV propulsion systems
* rotor aerodynamics
* aerospace system optimisation