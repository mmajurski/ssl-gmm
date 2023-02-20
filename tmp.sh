#!/bin/bash

source /home/mmajursk/miniconda3/etc/profile.d/conda.sh
conda activate gmm


sleep 1
python main.py --output-dirpath=./kmeans/supervised-00 --trainer=supervised --last-layer='kmeans'&
sleep 1
python main.py --output-dirpath=./kmeans/fixmatch-kmeans-00 --trainer=fixmatch --last-layer='kmeans' &
wait

sleep 1
python main.py --output-dirpath=./kmeans/supervised-01 --trainer=supervised --last-layer='kmeans' &
sleep 1
python main.py --output-dirpath=./kmeans/fixmatch-kmeans-01 --trainer=fixmatch --last-layer='kmeans' &
wait



#n=250
#c=1
#
## setup workers in the background
#sleep 1 &
#sleep 2 &
#
#for i in {0..10}; do
#
## --pseudo-label-method
## --pseudo-label-threshold
## --inference-method
#
## inference_method = {gmm, cauchy, softmax}
##pseudo_label_method = {sort_resp, sort_neum, filter_resp_sort_numerator, filter_resp_sort_resp, filter_resp_percentile_sort_neum}
##pseudo_label_threshold = {0.9, 0.95, 0.98, 0.99, 1.0}  # only apply this to those where 'filter' is in method
##
##for 'softmax', resp and neum are the same (i.e. just the logits)
#
#
#  wait -n
#  python main.py --output-filepath=./models/only-supervised-n${n}-c${c}-models/id-000${i} --num_labeled_datapoints=${n} --cluster_per_class=${c} --inference-method=softmax --strong_augmentation --disable-ssl &
#  sleep 0.2
#
#
#  inf="softmax"
#  for method in "sort_resp" "filter_resp_sort_resp" "filter_resp_percentile_sort_neum"; do
#
#      if [[ $method == *"filter"* ]]; then
#        for thres in "0.99" "0.98" "0.95" "0.9" "0.8"; do
#          wait -n
#          python main.py --output-filepath=./models/ssl-${inf}-method${method}-thres${thres}-n${n}-c${c}-models/id-000${i} --num_labeled_datapoints=${n} --pseudo-label-method=${method} --pseudo-label-threshold=${thres} --cluster_per_class=${c} --inference-method=${inf} --strong_augmentation &
#          sleep 0.2
#
#        done
#      else
#
#        wait -n
#        python main.py --output-filepath=./models/ssl-${inf}-method${method}-n${n}-c${c}-models/id-000${i} --num_labeled_datapoints=${n} --pseudo-label-method=${method} --cluster_per_class=${c} --inference-method=${inf} --strong_augmentation &
#        sleep 0.2
#      fi
#  done
#
#
#  for inf in "gmm" "cauchy"; do
#    for method in "sort_resp" "sort_neum" "filter_resp_sort_numerator" "filter_resp_sort_resp" "filter_resp_percentile_sort_neum"; do
#
#          if [[ $method == *"filter"* ]]; then
#            for thres in "0.99" "0.98" "0.95" "0.9" "0.8"; do
#              wait -n
#              python main.py --output-filepath=./models/ssl-${inf}-method${method}-thres${thres}-n${n}-c${c}-models/id-000${i} --num_labeled_datapoints=${n} --pseudo-label-method=${method} --pseudo-label-threshold=${thres} --cluster_per_class=${c} --inference-method=${inf} --strong_augmentation &
#              sleep 0.2
#
#            done
#          else
#            wait -n
#            python main.py --output-filepath=./models/ssl-${inf}-method${method}-n${n}-c${c}-models/id-000${i} --num_labeled_datapoints=${n} --pseudo-label-method=${method} --cluster_per_class=${c} --inference-method=${inf} --strong_augmentation &
#            sleep 0.2
#          fi
#
#    done
#  done
#done
#
#wait

