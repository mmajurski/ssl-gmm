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
#SBATCH --job-name=cmm
#SBATCH -o log-%N.%j.out
#SBATCH --time=128:0:0

# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


source /mnt/isgnas/home/mmajursk/miniconda3/etc/profile.d/conda.sh
conda activate gmm

#LAST_LAYER=$1
#MODEL_NB=$2
#PL_DETERM=$3
#PL_TARGET=$4
#LOSS_TERMS=$5
#
#python main.py --output-dirpath=./models-20230417/fixmatch-${LAST_LAYER}-${MODEL_NB}-pl${PL_DETERM}-pltgt${PL_TARGET}-loss${LOSS_TERMS} --trainer=fixmatch-gmm --last-layer=${LAST_LAYER} --pseudo-label-determination=${PL_DETERM} --pseudo-label-target-logits=${PL_TARGET} --loss-terms=${LOSS_TERMS}


MODEL_NB=$1
TAU=$2
EMA_FLAG=$3

if [ "$EMA_FLAG" -gt 0 ]; then
  python main.py --output-dirpath=./models-20230417/fixmatch-ema-${MODEL_NB}-T${TAU} --trainer=fixmatch --last-layer=fc --tau=${TAU} --use-ema
else
  python main.py --output-dirpath=./models-20230417/fixmatch-stock-${MODEL_NB}-T${TAU} --trainer=fixmatch --last-layer=fc --tau=${TAU}
fi





