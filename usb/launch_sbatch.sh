#!/bin/bash


for file in ./config/classic_cv/aagmm/cifar10/*.yaml
do
  # fn=$(basename -- "$file")
  # if [[ $fn == aagmm* ]]
  # then
    echo "Processing ${file}"
	  sbatch sbatch_script.sh ${file}
  # fi
done
