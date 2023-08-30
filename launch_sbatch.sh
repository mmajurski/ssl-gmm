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

MODELS_PER_JOB=1

# trainer='fixmatch'
# ll='aa_gmm_d1'
# emb_dim=0
# label_count=250
# for embd_constraint in 'none' 'l2' 'mean_covar'
# do
#   sbatch sbatch_script.sh ${ll} ${learning_rate} ${emb_dim} ${i} ${embd_constraint} ${trainer} ${label_count} ${MODELS_PER_JOB}
#   i=$((i+MODELS_PER_JOB))
# done

# ll='aa_gmm'
# for embd_constraint in 'none' 'l2' 'mean_covar'
# do
#   sbatch sbatch_script.sh ${ll} ${learning_rate} ${emb_dim} ${i} ${embd_constraint} ${trainer} ${label_count} ${MODELS_PER_JOB}
#   i=$((i+MODELS_PER_JOB))
# done
# embd_constraint='mean_covar'
# sbatch sbatch_script.sh ${ll} ${learning_rate} ${emb_dim} ${i} ${embd_constraint} ${trainer} ${label_count} ${MODELS_PER_JOB}
# i=$((i+MODELS_PER_JOB))

# ll='kmeans'
# for embd_constraint in 'none' 'l2' 'mean_covar'
# do
#   sbatch sbatch_script.sh ${ll} ${learning_rate} ${emb_dim} ${i} ${embd_constraint} ${trainer} ${label_count} ${MODELS_PER_JOB}
#   i=$((i+MODELS_PER_JOB))
# done

# trainer='fixmatch'
# embd_constraint='none'
# sbatch sbatch_script.sh 'fc' ${learning_rate} ${emb_dim} ${i} ${embd_constraint} ${trainer} ${label_count} ${MODELS_PER_JOB}
# i=$((i+MODELS_PER_JOB))



for mn in 0 1 2 3 4
do

  for emb_dim in 0
  do
    for label_count in 10 #250  # 1, 4, and 25 per class
    do
          # trainer='supervised'
          # embd_constraint='none'
          # sbatch sbatch_script.sh 'fc' ${learning_rate} ${emb_dim} ${i} ${embd_constraint} ${trainer} ${label_count} ${MODELS_PER_JOB}
          # i=$((i+MODELS_PER_JOB))

          trainer='fixmatch'
          embd_constraint='none'
          sbatch sbatch_script.sh 'fc' ${learning_rate} ${emb_dim} ${i} ${embd_constraint} ${trainer} ${label_count} ${MODELS_PER_JOB}
          i=$((i+MODELS_PER_JOB))

          for ll in "kmeans" "aa_gmm" "aa_gmm_d1"
          do
              for embd_constraint in 'none' 'l2' 'mean_covar'
              do
                trainer='fixmatch'
                sbatch sbatch_script.sh ${ll} ${learning_rate} ${emb_dim} ${i} ${embd_constraint} ${trainer} ${label_count} ${MODELS_PER_JOB}
                i=$((i+MODELS_PER_JOB))
              done
        done
      done
  done
done