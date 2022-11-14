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
#SBATCH --time=186:0:0

source /mnt/isgnas/home/mmajursk/anaconda3/etc/profile.d/conda.sh
conda activate gmm
#i=$1
#n=$2
#c=1


# --pseudo-label-method
# --pseudo-label-threshold
# --inference-method

# inference_method = {gmm, cauchy, softmax}
#pseudo_label_method = {sort_resp, sort_neum, filter_resp_sort_numerator, filter_resp_sort_resp, filter_resp_percentile_sort_neum}
#pseudo_label_threshold = {0.9, 0.95, 0.98, 0.99, 1.0}  # only apply this to those where 'filter' is in method
#
#for 'softmax', resp and neum are the same (i.e. just the logits)


# setup 3 workers in the background
sleep 1 &
sleep 2 &
sleep 3 &
sleep 4 &
sleep 5 &
sleep 6 &



for i in {0..9}; do

for n in 250 1000 4000; do

for c in 2 3 4 5; do

inf="softmax"
for method in "sort_resp" "filter_resp_sort_resp" "filter_resp_percentile_sort_neum"; do

    if [[ $method == *"filter"* ]]; then
      for thres in "0.99" "0.98" "0.95" "0.9"; do
        wait -n
        python main.py --output-filepath=./models/ssl-${inf}-method${method}-thres${thres}-n${n}-c${c}-models/id-000${i} --num_labeled_datapoints=${n} --pseudo-label-method=${method} --pseudo-label-threshold=${thres} --cluster_per_class=${c} --inference-method=${inf} &
        sleep 0.2

      done
    else

      wait -n
      python main.py --output-filepath=./models/ssl-${inf}-method${method}-n${n}-c${c}-models/id-000${i} --num_labeled_datapoints=${n} --pseudo-label-method=${method} --cluster_per_class=${c} --inference-method=${inf} &
      sleep 0.2
    fi
done


for inf in "gmm" "cauchy"; do
  for method in "sort_resp" "sort_neum" "filter_resp_sort_numerator" "filter_resp_sort_resp" "filter_resp_percentile_sort_neum"; do

        if [[ $method == *"filter"* ]]; then
          for thres in "0.99" "0.98" "0.95" "0.9"; do
            wait -n
            python main.py --output-filepath=./models/ssl-${inf}-method${method}-thres${thres}-n${n}-c${c}-models/id-000${i} --num_labeled_datapoints=${n} --pseudo-label-method=${method} --pseudo-label-threshold=${thres} --cluster_per_class=${c} --inference-method=${inf} &
            sleep 0.2

          done
        else
          wait -n
          python main.py --output-filepath=./models/ssl-${inf}-method${method}-n${n}-c${c}-models/id-000${i} --num_labeled_datapoints=${n} --pseudo-label-method=${method} --cluster_per_class=${c} --inference-method=${inf} &
          sleep 0.2
        fi

  done
done
done
done
done

wait

