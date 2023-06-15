#!/bin/bash


## find starting output directory
#MODEL_DIR="./models-20230612"
#A=($MODEL_DIR/id-*)
#HIGHEST_DIR="${A[-1]##*/}"
#HIGHEST_VAL=$(echo $HIGHEST_DIR | tr -dc '0-9')
#TRIM_HIGHEST=$(echo $HIGHEST_VAL | sed 's/^0*//')
#i=$((TRIM_HIGHEST+1))
#
#emb_count=16
#fc_count=0
#for mn in 0 1
#do
#    for ll in "fc" "aa_gmm" "aa_gmm_d1" "kmeans_layer"
#    do
#
##        if [ "$ll" == "fc" ]; then
##          lr=0.01
##        else
##          lr=0.001
##        fi
#
#
#        for lr in 0.01 0.001
#        do
#            for fc_count in 0
#            do
#                # no interleave
#                printf -v src "id-%08d" ${i}
#                sbatch sbatch_script.sh ${ll} ${lr} ${emb_count} ${fc_count} ${MODEL_DIR}/${src} 0
#                i=$((i+1))
#
#                # interleave
#                printf -v src "id-%08d" ${i}
#                sbatch sbatch_script.sh ${ll} ${lr} ${emb_count} ${fc_count} ${MODEL_DIR}/${src} 1
#                i=$((i+1))
#            done
#        done
#    done
#done

# find starting output directory
MODEL_DIR="./models-extrafc"
A=($MODEL_DIR/id-*)
HIGHEST_DIR="${A[-1]##*/}"
HIGHEST_VAL=$(echo $HIGHEST_DIR | tr -dc '0-9')
TRIM_HIGHEST=$(echo $HIGHEST_VAL | sed 's/^0*//')
i=$((TRIM_HIGHEST+1))

emb_count=16
fc_count=0
for mn in 0 1
do
    for ll in "kmeans_layer"
    do

        for lr in 0.01
        do
            for fc_count in 1 2
            do
                # no interleave
                printf -v src "id-%08d" ${i}
                sbatch sbatch_script.sh ${ll} ${lr} ${emb_count} ${fc_count} ${MODEL_DIR}/${src} 0
                i=$((i+1))

                # interleave
                printf -v src "id-%08d" ${i}
                sbatch sbatch_script.sh ${ll} ${lr} ${emb_count} ${fc_count} ${MODEL_DIR}/${src} 1
                i=$((i+1))

                # no interleave
                printf -v src "id-%08d" ${i}
                sbatch sbatch_script.sh ${ll} ${lr} ${emb_count} ${fc_count} ${MODEL_DIR}/${src} 0 0
                i=$((i+1))

                # interleave
                printf -v src "id-%08d" ${i}
                sbatch sbatch_script.sh ${ll} ${lr} ${emb_count} ${fc_count} ${MODEL_DIR}/${src} 1 1
                i=$((i+1))
            done
        done
    done
done