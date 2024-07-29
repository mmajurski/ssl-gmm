#!/bin/bash

ofp="./saved_models/comp_pl"
for file in ./config/comp_pl/*.yaml
do
  fn=$(basename -- "$file")
  if [[ $fn != fixmatch* ]]; then
    echo "Skipping ${file}"
    continue
  fi
  # then


  # if a folder with the same name as the yaml file exists in ofp, skip
   if [ -d "${ofp}/${fn%.*}" ]; then
      echo "Skipping [existing] ${file}"
      continue
    fi

    echo "Processing ${file}"
	  sbatch sbatch_script.sh ${file}
  # fi
done
