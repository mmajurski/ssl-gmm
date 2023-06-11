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



# find starting output directory
MODEL_DIR="./models-20230611"
A=($MODEL_DIR/id-*)
HIGHEST_DIR="${A[-1]##*/}"
HIGHEST_VAL=$(echo $HIGHEST_DIR | tr -dc '0-9')
TRIM_HIGHEST=$(echo $HIGHEST_VAL | sed 's/^0*//')
i=$((TRIM_HIGHEST+1))

emb_count=16
fc_count=0
for mn in 0 1
do
 for ll in "aa_gmm" "aa_gmm_d1" "kmeans_layer"
 do
    for optim in "sgd" "adamw"
    do

      if [ "$optim" == "sgd" ]; then
        lr=0.03
      else
        lr=0.0003
      fi


      # no interleave
      printf -v src "id-%08d" ${i}
        sbatch sbatch_script_gmm.sh ${ll} ${lr} ${emb_count} ${fc_count} ${MODEL_DIR}/${src} ${optim} 0
        i=$((i+1))

        # interleave
        printf -v src "id-%08d" ${i}
        sbatch sbatch_script_gmm.sh ${ll} ${lr} ${emb_count} ${fc_count} ${MODEL_DIR}/${src} ${optim} 1
        i=$((i+1))

#        for fc_count in 0 1 2
#        do
#          printf -v src "id-%08d" ${i}
#          sbatch sbatch_script_gmm.sh ${ll} ${lr} ${emb_count} ${fc_count} ${MODEL_DIR}/${src} ${optim}
#          i=$((i+1))
#        done
#      done
    done
 done
done

