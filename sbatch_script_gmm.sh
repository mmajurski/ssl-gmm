#!/bin/bash
# **************************
# MODIFY THESE OPTIONS

#SBATCH --partition=isg
#SBATCH --exclude=p100
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --nice
#SBATCH --oversubscribe
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --job-name=gmm
#SBATCH -o log-%N.%j.out
#SBATCH --time=128:0:0

# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


source /mnt/isgnas/home/mmajursk/miniconda3/etc/profile.d/conda.sh
conda activate gmm

LAST_LAYER=$1
MODEL_NB=$2
VAL_ACC=$3
EMA_FLAG=$4


if [ "$EMA_FLAG" -gt 0 ]; then
  python main.py --output-dirpath=./models-20230417/fixmatch-${LAST_LAYER}-${MODEL_NB}-valacc${VAL_ACC}-ema${EMA_FLAG} --trainer=fixmatch-gmm --last-layer=${LAST_LAYER} --optimizer=adamw --learning-rate=3e-4 --val-acc-term=${VAL_ACC} --use-ema
else
  python main.py --output-dirpath=./models-20230417/fixmatch-${LAST_LAYER}-${MODEL_NB}-valacc${VAL_ACC}-ema${EMA_FLAG} --trainer=fixmatch-gmm --last-layer=${LAST_LAYER} --optimizer=adamw --learning-rate=3e-4 --val-acc-term=${VAL_ACC}
fi


#MODEL_NB=$1
#TAU=$2
#TAU_METHOD=$3
#EMA_FLAG=$4
#
#if [ "$EMA_FLAG" -gt 0 ]; then
#  python main.py --output-dirpath=./models-fixmatch-baseline/fixmatch-ema-${MODEL_NB}-T${TAU}-TM${TAU_METHOD} --trainer=fixmatch --last-layer=fc --tau=${TAU} --tau-method=${TAU_METHOD} --use-ema
#else
#  python main.py --output-dirpath=./models-fixmatch-baseline/fixmatch-stock-${MODEL_NB}-T${TAU}-TM${TAU_METHOD} --trainer=fixmatch --last-layer=fc --tau=${TAU} --tau-method=${TAU_METHOD}
#fi





