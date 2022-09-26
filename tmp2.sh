#!/bin/bash
# **************************
# MODIFY THESE OPTIONS

#SBATCH --partition=heimdall
#SBATCH --nodes=1
#SBATCH --nice
#SBATCH --oversubscribe
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --job-name=gmm
#SBATCH -o log-%N.%j.out
#SBATCH --time=96:0:0

source /mnt/isgnas/home/mmajursk/anaconda3/etc/profile.d/conda.sh
conda activate gmm
i=$1
n=$2

c=2

python main.py --output-filepath=./models/ssl-resp-n${n}-c${c}-models/id-000${i} --num_labeled_datapoints=${n} --pseudo-label-percentile-threshold="resp" --cluster_per_class=${c} &
sleep 0.2

python main.py --output-filepath=./models/ssl-neum-n${n}-c${c}-models/id-000${i} --num_labeled_datapoints=${n} --pseudo-label-percentile-threshold="neum" --cluster_per_class=${c} &
sleep 0.2

# ***********************

python main.py --output-filepath=./models/ssl-resp-cauchy-n${n}-c${c}-models/id-000${i} --num_labeled_datapoints=${n} --pseudo-label-percentile-threshold="resp" --cluster_per_class=${c} --inference-method=cauchy &
sleep 0.2

python main.py --output-filepath=./models/ssl-neum-cauchy-n${n}-c${c}-models/id-000${i} --num_labeled_datapoints=${n} --pseudo-label-percentile-threshold="neum" --cluster_per_class=${c} --inference-method=cauchy &
sleep 0.2


# Baselines (no SSL)
for c in 2 4 8; do
    wait -n
    python main.py --output-filepath=./models/ssl-resp-n${n}-c${c}-models/id-000${i} --num_labeled_datapoints=${n} --pseudo-label-percentile-threshold="resp" --cluster_per_class=${c} &
    sleep 0.2

    wait -n
    python main.py --output-filepath=./models/ssl-neum-n${n}-c${c}-models/id-000${i} --num_labeled_datapoints=${n} --pseudo-label-percentile-threshold="neum" --cluster_per_class=${c} &
    sleep 0.2

    # ***********************

    wait -n
    python main.py --output-filepath=./models/ssl-resp-cauchy-n${n}-c${c}-models/id-000${i} --num_labeled_datapoints=${n} --pseudo-label-percentile-threshold="resp" --cluster_per_class=${c} --inference-method=cauchy &
    sleep 0.2

    wait -n
    python main.py --output-filepath=./models/ssl-neum-cauchy-n${n}-c${c}-models/id-000${i} --num_labeled_datapoints=${n} --pseudo-label-percentile-threshold="neum" --cluster_per_class=${c} --inference-method=cauchy &
    sleep 0.2


    # ***********************
    for p in "0.99" "0.98" "0.95" "0.9" "0.75"; do
      wait -n
      python main.py --output-filepath=./models/ssl-perc${p}-n${n}-c${c}-models/id-000${i} --num_labeled_datapoints=${n} --pseudo-label-percentile-threshold=${p} --cluster_per_class=${c} &
      sleep 0.2
    done

    # ***********************

    for p in "0.99" "0.98" "0.95" "0.9" "0.75"; do
      wait -n
      python main.py --output-filepath=./models/ssl-cauchy-perc${p}-n${n}-c${c}-models/id-000${i} --num_labeled_datapoints=${n} --pseudo-label-percentile-threshold=${p} --cluster_per_class=${c} --inference-method=cauchy &
      sleep 0.2
    done
done

wait

