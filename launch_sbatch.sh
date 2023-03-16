#!/bin/bash

# find the directory with largest value


sbatch sbatch_script.sh 0.9 fc
sbatch sbatch_script.sh 0.95 fc
sbatch sbatch_script.sh 0.99 fc

sbatch sbatch_script.sh 0.9 gmm
sbatch sbatch_script.sh 0.95 gmm
sbatch sbatch_script.sh 0.99 gmm

sbatch sbatch_script.sh 0.9 cauchy
sbatch sbatch_script.sh 0.95 cauchy
sbatch sbatch_script.sh 0.99 cauchy