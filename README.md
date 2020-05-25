# GSFE-REFINEMENT
## Table of Contents

- [Background](#background)
- [Usage](#usage)
- [Maintainers](#Maintainers)


## Background
protein refinement by generalized solvation free energy theory

## Usage
The code need python3.7, biopython (1.74), numpy (1.17.2), torch (1.2.0) and Java(openjdk version "1.8.0_191")

you can run the code by "python3 auto_opi_mutation_model770_sincos.py --PATH native_start --native_name 101m__native.pdb --decoy_name 101m__model.pdb --device cpu --L1_smooth_parameter 1.2 --ENRAOPY_W 1"
"--PATH" is the path of native structure and decoys.

"--native_name" is the parameter of native structure name,

"--decoy_name" is the parameter decoy name need to be refined, 

"--device" is the parameter of using cpu or gpu,

"--L1_smooth_parameter loss" is the parameter of L1_smooth item,

"--entropy_w"  is the parameter of entropy weight.


## Maintainers

[@15526876318](https://github.com/15526876318).
