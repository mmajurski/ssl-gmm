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

# find starting output directory
MODEL_DIR="./models-20230609"
A=($MODEL_DIR/id-*)
HIGHEST_DIR="${A[-1]##*/}"
HIGHEST_VAL=$(echo $HIGHEST_DIR | tr -dc '0-9')
TRIM_HIGHEST=$(echo $HIGHEST_VAL | sed 's/^0*//')
i=$((TRIM_HIGHEST+1))

for mn in 0 1
do
 for ll in "aa_gmm_d1" "kmeans_layer"
 do
    for lr in 0.05 0.01 0.001
    do

      for emb_count in 16 32 64
      do
        for fc_count in 1 2
        do

          #LAST_LAYER=$1
          #EMA_FLAG=$2
          #LEARNING_RATE=$3
          #EMBD_DIM=$4
          #PRE_FC=$5
          #MODEL_FP=$6
          printf -v src "id-%08d" ${i}
          sbatch sbatch_script_gmm.sh ${ll} ${ema} ${lr} ${emb_count} ${fc_count} ${MODEL_DIR}/${src}
          i=$((i+1))
        done
      done
    done
 done
done

