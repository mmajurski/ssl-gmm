#!/bin/bash

# find the directory with largest value


for n in 250 1000 4000; do
	for i in {0..4}; do
		sbatch tmp2.sh ${i} ${n}
	done
done
