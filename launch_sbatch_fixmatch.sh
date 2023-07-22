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

MODELS_PER_JOB=2
fc_count=0
for emb_dim in 16
do
  for label_count in 40 250  # 1, 4, and 25 per class
  do

    for mn in 0 1 2 3 4
    do
          trainer='fixmatch'
          embd_constraint='none'
          sbatch sbatch_script.sh 'fc' ${learning_rate} ${emb_dim} ${fc_count} ${i} ${embd_constraint} ${trainer} ${label_count} ${MODELS_PER_JOB} 0
          i=$((i+MODELS_PER_JOB))
      done
    done
done

