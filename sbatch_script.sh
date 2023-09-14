#!/bin/bash
# **************************
# MODIFY THESE OPTIONS

#SBATCH --partition=isg
#SBATCH --exclude=p100,quebec
#SBATCH --nodes=1
#SBATCH --oversubscribe
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --job-name=g40-ec
#SBATCH -o log-%N.%j.out
#SBATCH --time=96:0:0

# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


source /mnt/isgnas/home/mmajursk/miniconda3/etc/profile.d/conda.sh
conda activate gmm

#SBATCH --exclude=p100,quebec
#SBATCH --nodelist=oscar


LAST_LAYER=$1
EMBD_DIM=$2
START_RUN=$3
EMBD_CONSTRAINT=$4
NLABELS=$5
MODELS_PER_JOB=$6
OOD_PERC=$7
STRONG_PL_EMB_CONSTRAINT=$8
WEAK_PL_EMB_CONSTRAINT=$9




LEARNING_RATE=0.01
TRAINER='fixmatch'

echo "Model Number Starting PointCount = $START_RUN"
echo "Requested Model Count = $MODELS_PER_JOB"

root_output_directory="./models-emb-const"
if ! [ -d ${root_output_directory} ]; then
    mkdir ${root_output_directory}
fi



N_PARALLEL=1
gpustr=$(nvidia-smi --query-gpu=gpu_name --format=csv)
echo $gpustr
substr1="80GB"
substr2="40GB"
substr3="H100"
if [[ $gpustr == *"$substr1"* ]]; then
N_PARALLEL=2  # how many parallel trains to run on each job
elif [[ $gpustr == *"$substr2"* ]]; then
N_PARALLEL=2  # how many parallel trains to run on each job
elif [[ $gpustr == *"$substr3"* ]]; then
N_PARALLEL=2  # how many parallel trains to run on each job
else
N_PARALLEL=1  # how many parallel trains to run on each job
fi

echo "Training $N_PARALLEL models in parallel for this slurm job."

INDEX=$START_RUN
SUCCESS_COUNT=0
for i in $(seq $MODELS_PER_JOB); do
  if [ $i -gt $N_PARALLEL ]; then
    wait -n  # wait for the next job to terminate
    sc=$? # get status code from main
     if [ $sc -eq 0 ]; then
       SUCCESS_COUNT=$((SUCCESS_COUNT+1))
       echo "Successfully built $SUCCESS_COUNT models"
     fi
     if [ $SUCCESS_COUNT -ge $MODELS_PER_JOB ]; then
       exit 0
     fi
  fi

  MODEL_FP="${root_output_directory}/id-$(printf "%08d" ${INDEX})"

  # if [ $STRONG_PL_EMB_CONSTRAINT -gt 0 ] && [ $WEAK_PL_EMB_CONSTRAINT -gt 0 ]; then
  #   python main.py --output-dirpath=${MODEL_FP} --trainer=${TRAINER} --last-layer=${LAST_LAYER} --optimizer=sgd --learning-rate=${LEARNING_RATE} --embedding_dim=${EMBD_DIM} --embedding-constraint=${EMBD_CONSTRAINT} --num-labeled-datapoints=${NLABELS} --constrain-weak-pl-embedding --constrain-strong-pl-embedding --seed=4294967295 &

  # elif [ $STRONG_PL_EMB_CONSTRAINT -gt 0 ]; then
  #   python main.py --output-dirpath=${MODEL_FP} --trainer=${TRAINER} --last-layer=${LAST_LAYER} --optimizer=sgd --learning-rate=${LEARNING_RATE} --embedding_dim=${EMBD_DIM} --embedding-constraint=${EMBD_CONSTRAINT} --num-labeled-datapoints=${NLABELS} --constrain-strong-pl-embedding --seed=4294967295 &

  # elif [ $WEAK_PL_EMB_CONSTRAINT -gt 0 ]; then
  #   python main.py --output-dirpath=${MODEL_FP} --trainer=${TRAINER} --last-layer=${LAST_LAYER} --optimizer=sgd --learning-rate=${LEARNING_RATE} --embedding_dim=${EMBD_DIM} --embedding-constraint=${EMBD_CONSTRAINT} --num-labeled-datapoints=${NLABELS} --constrain-weak-pl-embedding --seed=4294967295 &

  # else
  #   python main.py --output-dirpath=${MODEL_FP} --trainer=${TRAINER} --last-layer=${LAST_LAYER} --optimizer=sgd --learning-rate=${LEARNING_RATE} --embedding_dim=${EMBD_DIM} --embedding-constraint=${EMBD_CONSTRAINT} --num-labeled-datapoints=${NLABELS} --seed=4294967295 &

  # fi

  python main.py --output-dirpath=${MODEL_FP} --trainer=${TRAINER} --last-layer=${LAST_LAYER} --optimizer=sgd --learning-rate=${LEARNING_RATE} --embedding_dim=${EMBD_DIM} --embedding-constraint=${EMBD_CONSTRAINT} --num-labeled-datapoints=${NLABELS} --ood_p=${OOD_PERC} &
  #python main.py --output-dirpath=${MODEL_FP} --trainer=${TRAINER} --last-layer=${LAST_LAYER} --optimizer=sgd --learning-rate=${LEARNING_RATE} --embedding_dim=${EMBD_DIM} --embedding-constraint=${EMBD_CONSTRAINT} --num-labeled-datapoints=${NLABELS} --arch="wide_resnet28-8" --num-classes=100 --ood_p=${OOD_PERC} &

  INDEX=$((INDEX+1))
  sleep 4  # separate launches by 1 second minimum
done


# wait for all of the runs to complete before exiting
wait
