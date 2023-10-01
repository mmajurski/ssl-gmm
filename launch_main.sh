#!/bin/bash

source /home/mmajursk/miniconda3/etc/profile.d/conda.sh
conda activate gmm

# find starting output directory
MODEL_DIR="./models"
A=($MODEL_DIR/id-*)
HIGHEST_DIR="${A[-1]##*/}"
HIGHEST_VAL=$(echo $HIGHEST_DIR | tr -dc '0-9')
TRIM_HIGHEST=$(echo $HIGHEST_VAL | sed 's/^0*//')
i=$((TRIM_HIGHEST+1))


LEARNING_RATE=0.01
for mn in 0 1
do
    for NLABELS in 400  # 1, 4, and 25 per class
    do
#         trainer='supervised'
#         embd_constraint='none'
#         sbatch sbatch_script.sh 'fc' ${learning_rate} ${emb_dim} ${i} ${embd_constraint} ${trainer} ${label_count} ${MODELS_PER_JOB}
#         i=$((i+MODELS_PER_JOB))

         # trainer='fixmatch'
         # embd_constraint='none'
         # sbatch sbatch_script.sh 'fc' ${learning_rate} ${emb_dim} ${i} ${embd_constraint} ${trainer} ${label_count} ${MODELS_PER_JOB}
         # i=$((i+MODELS_PER_JOB))

          for LAST_LAYER in "kmeans" "aa_gmm_d1"  #  "aa_gmm"
          do
              for EMBD_CONSTRAINT in 'mean_covar'  #'none' 'l2'
              do
                TRAINER='fixmatch'
                MODEL_FP="${MODEL_DIR}/id-$(printf "%08d" ${i})"
                python main.py --output-dirpath=${MODEL_FP} --trainer=${TRAINER} --last-layer=${LAST_LAYER} --optimizer=sgd --learning-rate=${LEARNING_RATE} --embedding-constraint=${EMBD_CONSTRAINT} --num-labeled-datapoints=${NLABELS} --seed=4694767885787126309

#                sbatch sbatch_script.sh ${ll} ${learning_rate} ${emb_dim} ${i} ${embd_constraint} ${trainer} ${label_count} ${MODELS_PER_JOB}
                i=$((i+1))
              done
        done
      done
done