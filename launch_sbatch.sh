#!/bin/bash




##MODEL_NB=$1
##TAU=$2
##EMA_FLAG=$3
#
#for mn in 0 1 2 3 4 5
#do
#
#sbatch sbatch_script.sh ${mn} 1.0 0
#sbatch sbatch_script.sh ${mn} 1.0 1
#
#sbatch sbatch_script.sh ${mn} 0.95 0
#sbatch sbatch_script.sh ${mn} 0.95 1
#
#sbatch sbatch_script.sh ${mn} 0.9 0
#sbatch sbatch_script.sh ${mn} 0.9 1
#
#sbatch sbatch_script.sh ${mn} 0.8 0
#sbatch sbatch_script.sh ${mn} 0.8 1
#
#sbatch sbatch_script.sh ${mn} 0.7 0
#sbatch sbatch_script.sh ${mn} 0.7 1
#
#sbatch sbatch_script.sh ${mn} 0.6 0
#sbatch sbatch_script.sh ${mn} 0.6 1
#
#sbatch sbatch_script.sh ${mn} 0.5 0
#sbatch sbatch_script.sh ${mn} 0.5 1
#
#sbatch sbatch_script.sh ${mn} 0.4 0
#sbatch sbatch_script.sh ${mn} 0.4 1
#
#done



#LAST_LAYER=$1
#MODEL_NB=$2
#PL_DETERM=$3
#PL_TARGET=$4
#LOSS_TERMS=$5
#VAL_ACC=$6

for mn in 0 1 3
do

  for ll in "aa_gmm" "aa_gmm_d1"
  do

    for loss_term in "gmm+cluster" "cmm+cluster" "gmm+cmm+cluster"
    do

      for va in "gmm" "cmm" "gmmcmm"
      do

        sbatch sbatch_script.sh ${ll} ${mn} gmm gmm ${loss_term} ${va}
        sbatch sbatch_script.sh ${ll} ${mn} gmm cmm ${loss_term} ${va}
        sbatch sbatch_script.sh ${ll} ${mn} cmm cmm ${loss_term} ${va}

      done
    done
  done
done
