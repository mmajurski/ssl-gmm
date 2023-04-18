#!/bin/bash




#MODEL_NB=$1
#TAU=$2
#EMA_FLAG=$3

sbatch sbatch_script.sh 0 1.0 0
sbatch sbatch_script.sh 0 1.0 1
sbatch sbatch_script.sh 1 1.0 0
sbatch sbatch_script.sh 1 1.0 1

sbatch sbatch_script.sh 0 0.95 0
sbatch sbatch_script.sh 0 0.95 1
sbatch sbatch_script.sh 1 0.95 0
sbatch sbatch_script.sh 1 0.95 1

sbatch sbatch_script.sh 0 0.9 0
sbatch sbatch_script.sh 0 0.9 1
sbatch sbatch_script.sh 1 0.9 0
sbatch sbatch_script.sh 1 0.9 1

sbatch sbatch_script.sh 0 0.8 0
sbatch sbatch_script.sh 0 0.8 1
sbatch sbatch_script.sh 1 0.8 0
sbatch sbatch_script.sh 1 0.8 1

sbatch sbatch_script.sh 0 0.7 0
sbatch sbatch_script.sh 0 0.7 1
sbatch sbatch_script.sh 1 0.7 0
sbatch sbatch_script.sh 1 0.7 1

sbatch sbatch_script.sh 0 0.6 0
sbatch sbatch_script.sh 0 0.6 1
sbatch sbatch_script.sh 1 0.6 0
sbatch sbatch_script.sh 1 0.6 1

sbatch sbatch_script.sh 0 0.5 0
sbatch sbatch_script.sh 0 0.5 1
sbatch sbatch_script.sh 1 0.5 0
sbatch sbatch_script.sh 1 0.5 1

sbatch sbatch_script.sh 0 0.4 0
sbatch sbatch_script.sh 0 0.4 1
sbatch sbatch_script.sh 1 0.4 0
sbatch sbatch_script.sh 1 0.4 1


#LAST_LAYER=$1
#MODEL_NB=$2
#PL_DETERM=$3
#PL_TARGET=$4
#LOSS_TERMS=$5

#sbatch sbatch_script.sh aa_gmm_d1 0 gmm gmm gmm+cluster
#sbatch sbatch_script.sh aa_gmm_d1 1 gmm gmm gmm+cluster
#sbatch sbatch_script.sh aa_gmm_d1 0 gmm gmm cmm+cluster
#sbatch sbatch_script.sh aa_gmm_d1 1 gmm gmm cmm+cluster
#sbatch sbatch_script.sh aa_gmm_d1 0 gmm gmm gmm+cmm+cluster
#sbatch sbatch_script.sh aa_gmm_d1 1 gmm gmm gmm+cmm+cluster
#
#sbatch sbatch_script.sh aa_gmm_d1 0 gmm cmm gmm+cluster
#sbatch sbatch_script.sh aa_gmm_d1 1 gmm cmm gmm+cluster
#sbatch sbatch_script.sh aa_gmm_d1 0 gmm cmm cmm+cluster
#sbatch sbatch_script.sh aa_gmm_d1 1 gmm cmm cmm+cluster
#sbatch sbatch_script.sh aa_gmm_d1 0 gmm cmm gmm+cmm+cluster
#sbatch sbatch_script.sh aa_gmm_d1 1 gmm cmm gmm+cmm+cluster
#
#sbatch sbatch_script.sh aa_gmm_d1 0 cmm cmm gmm+cluster
#sbatch sbatch_script.sh aa_gmm_d1 1 cmm cmm gmm+cluster
#sbatch sbatch_script.sh aa_gmm_d1 0 cmm cmm cmm+cluster
#sbatch sbatch_script.sh aa_gmm_d1 1 cmm cmm cmm+cluster
#sbatch sbatch_script.sh aa_gmm_d1 0 cmm cmm gmm+cmm+cluster
#sbatch sbatch_script.sh aa_gmm_d1 1 cmm cmm gmm+cmm+cluster
#
#sbatch sbatch_script.sh aa_gmm 0 gmm gmm gmm+cluster
#sbatch sbatch_script.sh aa_gmm 1 gmm gmm gmm+cluster
#sbatch sbatch_script.sh aa_gmm 0 gmm gmm cmm+cluster
#sbatch sbatch_script.sh aa_gmm 1 gmm gmm cmm+cluster
#sbatch sbatch_script.sh aa_gmm 0 gmm gmm gmm+cmm+cluster
#sbatch sbatch_script.sh aa_gmm 1 gmm gmm gmm+cmm+cluster
#
#sbatch sbatch_script.sh aa_gmm 0 gmm cmm gmm+cluster
#sbatch sbatch_script.sh aa_gmm 1 gmm cmm gmm+cluster
#sbatch sbatch_script.sh aa_gmm 0 gmm cmm cmm+cluster
#sbatch sbatch_script.sh aa_gmm 1 gmm cmm cmm+cluster
#sbatch sbatch_script.sh aa_gmm 0 gmm cmm gmm+cmm+cluster
#sbatch sbatch_script.sh aa_gmm 1 gmm cmm gmm+cmm+cluster
#
#sbatch sbatch_script.sh aa_gmm 0 cmm cmm gmm+cluster
#sbatch sbatch_script.sh aa_gmm 1 cmm cmm gmm+cluster
#sbatch sbatch_script.sh aa_gmm 0 cmm cmm cmm+cluster
#sbatch sbatch_script.sh aa_gmm 1 cmm cmm cmm+cluster
#sbatch sbatch_script.sh aa_gmm 0 cmm cmm gmm+cmm+cluster
#sbatch sbatch_script.sh aa_gmm 1 cmm cmm gmm+cmm+cluster