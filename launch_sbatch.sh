#!/bin/bash

#LAST_LAYER=$1
#MODEL_NB=$2
#PL_DETERM=$3
#PL_TARGET=$4
#LOSS_TERMS=$5

sbatch sbatch_script.sh aa_gmm_d1 0 gmm gmm gmm+cluster
sbatch sbatch_script.sh aa_gmm_d1 1 gmm gmm gmm+cluster
sbatch sbatch_script.sh aa_gmm_d1 0 gmm gmm cmm+cluster
sbatch sbatch_script.sh aa_gmm_d1 1 gmm gmm cmm+cluster
sbatch sbatch_script.sh aa_gmm_d1 0 gmm gmm gmm+cmm+cluster
sbatch sbatch_script.sh aa_gmm_d1 1 gmm gmm gmm+cmm+cluster

sbatch sbatch_script.sh aa_gmm_d1 0 gmm cmm gmm+cluster
sbatch sbatch_script.sh aa_gmm_d1 1 gmm cmm gmm+cluster
sbatch sbatch_script.sh aa_gmm_d1 0 gmm cmm cmm+cluster
sbatch sbatch_script.sh aa_gmm_d1 1 gmm cmm cmm+cluster
sbatch sbatch_script.sh aa_gmm_d1 0 gmm cmm gmm+cmm+cluster
sbatch sbatch_script.sh aa_gmm_d1 1 gmm cmm gmm+cmm+cluster

sbatch sbatch_script.sh aa_gmm_d1 0 cmm cmm gmm+cluster
sbatch sbatch_script.sh aa_gmm_d1 1 cmm cmm gmm+cluster
sbatch sbatch_script.sh aa_gmm_d1 0 cmm cmm cmm+cluster
sbatch sbatch_script.sh aa_gmm_d1 1 cmm cmm cmm+cluster
sbatch sbatch_script.sh aa_gmm_d1 0 cmm cmm gmm+cmm+cluster
sbatch sbatch_script.sh aa_gmm_d1 1 cmm cmm gmm+cmm+cluster

sbatch sbatch_script.sh aa_gmm 0 gmm gmm gmm+cluster
sbatch sbatch_script.sh aa_gmm 1 gmm gmm gmm+cluster
sbatch sbatch_script.sh aa_gmm 0 gmm gmm cmm+cluster
sbatch sbatch_script.sh aa_gmm 1 gmm gmm cmm+cluster
sbatch sbatch_script.sh aa_gmm 0 gmm gmm gmm+cmm+cluster
sbatch sbatch_script.sh aa_gmm 1 gmm gmm gmm+cmm+cluster

sbatch sbatch_script.sh aa_gmm 0 gmm cmm gmm+cluster
sbatch sbatch_script.sh aa_gmm 1 gmm cmm gmm+cluster
sbatch sbatch_script.sh aa_gmm 0 gmm cmm cmm+cluster
sbatch sbatch_script.sh aa_gmm 1 gmm cmm cmm+cluster
sbatch sbatch_script.sh aa_gmm 0 gmm cmm gmm+cmm+cluster
sbatch sbatch_script.sh aa_gmm 1 gmm cmm gmm+cmm+cluster

sbatch sbatch_script.sh aa_gmm 0 cmm cmm gmm+cluster
sbatch sbatch_script.sh aa_gmm 1 cmm cmm gmm+cluster
sbatch sbatch_script.sh aa_gmm 0 cmm cmm cmm+cluster
sbatch sbatch_script.sh aa_gmm 1 cmm cmm cmm+cluster
sbatch sbatch_script.sh aa_gmm 0 cmm cmm gmm+cmm+cluster
sbatch sbatch_script.sh aa_gmm 1 cmm cmm gmm+cmm+cluster