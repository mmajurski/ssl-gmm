#!/bin/bash


for file in ./config/classic_cv/aagmm/cifar10/*.yaml
do
  # if file starts with 'kmeans' then run
  fn=$(basename -- "$file")
  if [[ $fn == linear* ]]
  then
    echo "Processing ${file}"
	sbatch sbatch_script.sh ${file}
  fi
done
