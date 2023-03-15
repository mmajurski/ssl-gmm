#!/bin/bash

source /home/mmajursk/miniconda3/etc/profile.d/conda.sh
conda activate gmm

python main.py --output-dirpath=./models-20230314/fixmatch-fc-01 --trainer=fixmatch --last-layer=fc --pseudo-label-threshold=0.95
python main.py --output-dirpath=./models-20230314/fixmatch-fc-02 --trainer=fixmatch --last-layer=fc --pseudo-label-threshold=0.95
python main.py --output-dirpath=./models-20230314/fixmatch-gmm-01 --trainer=fixmatch --last-layer=gmm --pseudo-label-threshold=0.95
python main.py --output-dirpath=./models-20230314/fixmatch-gmm-02 --trainer=fixmatch --last-layer=gmm --pseudo-label-threshold=0.95

python main.py --output-dirpath=./models-20230314/fixmatch-fc-01 --trainer=fixmatch --last-layer=fc --pseudo-label-threshold=0.9
python main.py --output-dirpath=./models-20230314/fixmatch-fc-02 --trainer=fixmatch --last-layer=fc --pseudo-label-threshold=0.9
python main.py --output-dirpath=./models-20230314/fixmatch-gmm-01 --trainer=fixmatch --last-layer=gmm --pseudo-label-threshold=0.9
python main.py --output-dirpath=./models-20230314/fixmatch-gmm-02 --trainer=fixmatch --last-layer=gmm --pseudo-label-threshold=0.9

#MODELS_PER_JOB=2
#MAX_MODEL_ATTEMPT_COUNT=1000
#N_PARALLEL=2
#INDEX=1
#SUCCESS_COUNT=0
#
#for i in $(seq $MODELS_PER_JOB); do
#  python main.py --output-dirpath=./models-20230304/id-${INDEX} --trainer=fixmatch --last-layer=gmm --supervised-pretrain
#  INDEX=$((INDEX+1))
#done

#for i in $(seq $MAX_MODEL_ATTEMPT_COUNT); do
#  if [ $i -gt $N_PARALLEL ]; then
#    wait -n  # wait for the next job to terminate
#    sc=$? # get status code from main
#     if [ $sc -eq 0 ]; then
#       SUCCESS_COUNT=$((SUCCESS_COUNT+1))
#       echo "Successfully built $SUCCESS_COUNT models"
#     fi
#  fi
#
# if [ $SUCCESS_COUNT -lt $MODELS_PER_JOB ]; then
#   python main.py --output-dirpath=./kmeans/id-${INDEX} --trainer='fixmatch' --last-layer='kmeans' --supervised-pretrain &
#   INDEX=$((INDEX+1))
#   sleep 1  # separate launches by 1 second minimum
# fi
#done
#
#
## wait for all of the runs to complete before exiting
#wait



