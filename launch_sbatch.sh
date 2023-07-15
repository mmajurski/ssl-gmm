#!/bin/bash


# find starting output directory
MODEL_DIR="./models"
A=($MODEL_DIR/id-*)
HIGHEST_DIR="${A[-1]##*/}"
HIGHEST_VAL=$(echo $HIGHEST_DIR | tr -dc '0-9')
TRIM_HIGHEST=$(echo $HIGHEST_VAL | sed 's/^0*//')
i=$((TRIM_HIGHEST+1))

#LAST_LAYER=$1
#LEARNING_RATE=$2
#EMBD_DIM=$3
#PRE_FC=$4
#MODEL_FP=$5
#EMBD_CONSTRAINT=$6
#TRAINER=$7
#NLABELS=$8



learning_rate=0.01
#for mn in 0 1 2
#do
#    for emb_dim in 16
#    do
#      for label_count in 1 40 250
#      do
#          for fc_count in 0 1
#          do
#
#              printf -v src "id-%08d" ${i}
#              embd_constraint='none'
#              trainer='supervised'
#              #sbatch sbatch_script.sh 'fc' ${learning_rate} ${emb_dim} ${fc_count} ${MODEL_DIR}/${src} ${embd_constraint} ${trainer} ${label_count}
#              i=$((i+1))
#
#              for ll in "kmeans" "aa_gmm" "aa_gmm_d1" "aa_cmm" "aa_cmm_d1"
#              do
#                  for embd_constraint in 'none' 'mean_covar' 'gauss_moment'
#                  do
#                    printf -v src "id-%08d" ${i}
#                    trainer='fixmatch'
#                    #sbatch sbatch_script.sh ${ll} ${learning_rate} ${emb_dim} ${fc_count} ${MODEL_DIR}/${src} ${embd_constraint} ${trainer} ${label_count}
#                    i=$((i+1))
#                  done
#              done
#          done
#        done
#    done
#done

MODELS_PER_JOB=3
for emb_dim in 16
do
  for label_count in 10 40 250  # 1, 4, and 25 per class
  do
      for fc_count in 0 1
      do

          embd_constraint='none'
          trainer='supervised'
          sbatch sbatch_script.sh 'fc' ${learning_rate} ${emb_dim} ${fc_count} ${i} ${embd_constraint} ${trainer} ${label_count} ${MODELS_PER_JOB}
          i=$((i+MODELS_PER_JOB))

#          for ll in "kmeans" "aa_gmm" "aa_gmm_d1" "aa_cmm" "aa_cmm_d1"
#          do
#              for embd_constraint in 'none' 'mean_covar' 'gauss_moment'
#              do
#                trainer='fixmatch'
#                sbatch sbatch_script.sh ${ll} ${learning_rate} ${emb_dim} ${fc_count} ${i} ${embd_constraint} ${trainer} ${label_count} ${MODELS_PER_JOB}
#                i=$((i+MODELS_PER_JOB))
#              done
#          done
      done
    done
done
