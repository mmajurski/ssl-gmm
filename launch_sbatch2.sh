#!/bin/bash

# find the directory with largest value


for n in 250 1000 4000; do
	for i in {7..12}; do
		sbatch tmp3.sh ${i} ${n}
	done
done
