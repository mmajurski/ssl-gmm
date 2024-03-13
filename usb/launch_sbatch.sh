#!/bin/bash


for file in ./config/classic_cv/aagmm/*.yaml
do
	fn=$(basename $file)
	echo "Processing ${fn}"
	sbatch sbatch_script.sh ${fn}
done
