Optimisation of coaxial propellers for a given aircraft in a given flight state.

- aircraft input: 
 - aircraft weight: 650kg 
 - N_rotors (coaxial rotors): 8
 - N_blades: 2

- design flight state input:
 - ISA altitude: 500m
 - design climb speed: 3 m/s

- Inputs for final report:
 - NACA airfoil: 2412
 - Disk loading: 160 N/m²  
   (calculated for each rotor --> distribute loading on 2xNrotor)
 - design blade loading: 0.1 
 - relation R_upper/R_lower = 1

- optimisation goals: 
 - minimum power in the given flight state at the target thrust

- twist and chord distribution must be mapped to distributions (number of sections must be able to be changed)

- optimisation parameters:
- twist:
   -  twist independent for both rotors
   - non-linear twist distribution
- chord:
   - chord independent distribution: 4 parameter cubic spline
   - same chord distribution for both propellers (to reduce optimisation complexity)
- tip speed, tip mach number start value = 0.4

- optimisation output:
 - optimised propeller definition

Additional parameter sweep of aircraft power demand vs. climb speed (+10 m/s to -10m/s in 1m/s steps)
- For this, you have to iteratively calculate (trim) rpm to match the thrust at the given flight speed.

Additional requirements:
Do NOT use absolute file paths
Only read files at the beginning and write files at the end
DO NOT read/write inside functions, etc.
No global variables
Don't hardcode things that do not NEED to be hard-coded (like the number of sections, or the root cutout)
30 radial stations

Final submission:
Turn in your final code for optimisation of coaxial propellers in hover and climb flight
--> Code must be commented to include explanations of all steps, including theory
Report that includes:
- Description of propeller optimisation/design loop (small diagram)
- Description of design case
- Results:
  - Plot of twist and chord distribution for both propellers
   (--> Explanation for more complex distributions)
  - Plot of inflow angles and produced thrust along the radius for both propellers
  - Power vs. climb speed (Including AND excluding climb power)
  

Grading:
Correctness of implementation
More complex twist/chord distributions or more optimisation parameters result in a higher grade
--> have to be explained well
Formatting, correctness of the report and plots

Also:
Use a root cutout of 0.1R and don't use the section data we provided for the group propellers (these don't have to be 3D printed). Instead of the neuralfoil, we can use the data of the NACA 2412
