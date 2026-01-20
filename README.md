Optimization of coaxial propellers for a given aircraft in a given flight state.

The code should do the following steps:

aircraft input: Aircraft weight, number of propellers, coaxial (yes or no), Tip Mach number
design flight state input: Climb speed, RPM, altitude
optimization goals: minimum power in given flight state at target thrust
twist and chord distribution must be mapped to distributions (number of sections must be able to be changed)
optimization parameters (minimum):
- twist:
   - twist indipendent for both rotors
   - linear twist distribution
- chord:
   - chord distribution: 4 parameter cubic spline
   - same chord distribution for both propellers (to reduce optimization complexity)
- One additional optimization parameter: (Only one of these)
  - disk loading (same for both rotors) (use area of both rotors for calculation)
  - tip speed
  - relation of disk area between upper and lower rotor
optimization output:
- optimized propeller definition
Additional parameter sweep of aircraft power demand vs. climb speed (+10 m/s to -10m/s in 1m/s steps)
- For this you have to iteratively calculate (trim) rpm to match the thrust at the given flight speed.

Additional requirements:
Do NOT use absolute file paths
Only read files at the beginning and write files at the end
DO NOT read/write inside functions etc.
No global variables
Don't hardcode things that do not NEED to be hard-coded (like number of sections, or root cutout)
30 radial stations


Final submission:

Turn in your final code for opimization of coaxial propellers in hover and climb flight
--> Code must be commented to include explanations of all steps incl theory
Report that includes:
- Description of propeller optimization/design loop (small diagram)
- Description of design case
- Results:
  - Plot of twist and chord distribution for both propellers
   (--> Explanation for more complex distributions)
  - Plot of inflow angles and produced thrust along the radius for both propellers
  - Power vs. climb speed (Including AND excluding climb power)
  
Inputs from us to you:
Aircraft weight
Number of rotors
Altitude
design climb speed
NACA airfoil
disk loading (DL), tip mach number (Mtip), design blade loading 
your additional optimization parameter (if Mtip or DL, use the given value as a START value)

Grading:
Correctness of implementation
More complex twist/chord distributions or more optimization paramaters result in higher grade
--> have to be explained well
Formatting, correctness of report and plots

Also:
Use root cutout = 0.1R and dont use the section data we provided for the group propellers (these dont have to be 3d printed)
