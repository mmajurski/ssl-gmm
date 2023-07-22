#!/bin/bash


# find starting output directory
MODEL_DIR="./models"
A=($MODEL_DIR/id-*)
HIGHEST_DIR="${A[-1]##*/}"
HIGHEST_VAL=$(echo $HIGHEST_DIR | tr -dc '0-9')
TRIM_HIGHEST=$(echo $HIGHEST_VAL | sed 's/^0*//')
i=$((TRIM_HIGHEST+1))



learning_rate=0.01

MODELS_PER_JOB=2
for emb_dim in 16
do
  for label_count in 40 250  # 1, 4, and 25 per class
  do
      for fc_count in 0 1
      do

#          trainer='supervised'
#          embd_constraint='none'
#          sbatch sbatch_script.sh 'fc' ${learning_rate} ${emb_dim} ${fc_count} ${i} ${embd_constraint} ${trainer} ${label_count} ${MODELS_PER_JOB} "1724865484"
#          i=$((i+MODELS_PER_JOB))
#
#          trainer='fixmatch'
#          embd_constraint='none'
#          sbatch sbatch_script.sh 'fc' ${learning_rate} ${emb_dim} ${fc_count} ${i} ${embd_constraint} ${trainer} ${label_count} ${MODELS_PER_JOB} "1724865484"
#          i=$((i+MODELS_PER_JOB))

          for ll in "kmeans" "aa_gmm" "aa_gmm_d1"
          do
#              for embd_constraint in 'none' 'l2' 'mean_covar' 'gauss_moment'
              for embd_constraint in 'mean_covar2' 'gauss_moment'
              do
                trainer='fixmatch'
                sbatch sbatch_script.sh ${ll} ${learning_rate} ${emb_dim} ${fc_count} ${i} ${embd_constraint} ${trainer} ${label_count} ${MODELS_PER_JOB} "1724865484"
                i=$((i+MODELS_PER_JOB))
              done
          done
      done
    done
done
