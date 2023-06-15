#!/bin/bash

source /home/mmajursk/miniconda3/etc/profile.d/conda.sh
conda activate gmm

# find starting output directory
MODEL_DIR="./models-20230615"
A=($MODEL_DIR/id-*)
HIGHEST_DIR="${A[-1]##*/}"
HIGHEST_VAL=$(echo $HIGHEST_DIR | tr -dc '0-9')
TRIM_HIGHEST=$(echo $HIGHEST_VAL | sed 's/^0*//')
i=$((TRIM_HIGHEST+1))

for mn in 0 1 2
do
    for ll in "aa_gmm" "aa_gmm_d1"
    do

        for lr in 0.01
        do

            for emb_count in 8 16 32
            do

                printf -v src "id-%08d" ${i}
                python main.py --output-dirpath=${MODEL_DIR}/${src} --trainer=fixmatch --last-layer=${ll} --optimizer=sgd --learning-rate=${lr} --embedding_dim=${emb_count} --nprefc=0 --use_tanh=0
                i=$((i+1))
            done
        done
    done
done