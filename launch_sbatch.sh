#!/bin/bash




## MODEL_NB=$1
## TAU=$2
## TAU_METHOD=$3
## EMA_FLAG=$4
#
#for mn in 0 1 2
#do
#
#  for ema in 0 1
#  do
#
#    # run tau 1.0, which only needs fixmatch, not mixmatch tau method
#    sbatch sbatch_script.sh ${mn} 1.0 fixmatch ${ema}
#
#    for tau in 0.95 0.9 0.8 0.7 0.6 0.5
#    do
#      sbatch sbatch_script.sh ${mn} ${tau} fixmatch ${ema}
#      sbatch sbatch_script.sh ${mn} ${tau} mixmatch ${ema}
#    done
#  done
#done


#
#LAST_LAYER=$1
#MODEL_NB=$2
#VAL_ACC=$3
ema=0

for mn in 0 1 2 3 4
do
 for ll in "aa_gmm" "aa_gmm_d1" "kmeans_cmm" "kmeans_layer"
 do
     sbatch sbatch_script_gmm.sh ${ll} ${mn} ${ema}
 done
done

