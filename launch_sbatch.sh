#!/bin/bash


# find starting output directory
MODEL_DIR="./models-cf10"
A=($MODEL_DIR/id-*)
HIGHEST_DIR="${A[-1]##*/}"
HIGHEST_VAL=$(echo $HIGHEST_DIR | tr -dc '0-9')
TRIM_HIGHEST=$(echo $HIGHEST_VAL | sed 's/^0*//')
i=$((TRIM_HIGHEST+1))



MODELS_PER_JOB=1

# Seeds
# 3474173998
# 273230791
# 3586106167
# 1325645050
# 2564231920


for mn in 0
do

  for emb_dim in 0 8 32
  do
    for label_count in 40 250
    do

		      sbatch sbatch_script.sh 'fc' ${emb_dim} ${i} 'none' ${label_count} ${MODELS_PER_JOB} 3474173998
          i=$((i+MODELS_PER_JOB))
          sbatch sbatch_script.sh 'fc' ${emb_dim} ${i} 'none' ${label_count} ${MODELS_PER_JOB} 273230791
          i=$((i+MODELS_PER_JOB))
          sbatch sbatch_script.sh 'fc' ${emb_dim} ${i} 'none' ${label_count} ${MODELS_PER_JOB} 3586106167
          i=$((i+MODELS_PER_JOB))
          sbatch sbatch_script.sh 'fc' ${emb_dim} ${i} 'none' ${label_count} ${MODELS_PER_JOB} 1325645050
          i=$((i+MODELS_PER_JOB))
          sbatch sbatch_script.sh 'fc' ${emb_dim} ${i} 'none' ${label_count} ${MODELS_PER_JOB} 2564231920
          i=$((i+MODELS_PER_JOB))

          for ll in "kmeans" "aa_gmm"
          do
              for embd_constraint in 'none' 'l2' 'mean_covar' #'gauss_moment3' 'gauss_moment4'
              do

                sbatch sbatch_script.sh ${ll} ${emb_dim} ${i} ${embd_constraint} ${label_count} ${MODELS_PER_JOB} 3474173998
                i=$((i+MODELS_PER_JOB))
                sbatch sbatch_script.sh ${ll} ${emb_dim} ${i} ${embd_constraint} ${label_count} ${MODELS_PER_JOB} 273230791
                i=$((i+MODELS_PER_JOB))
                sbatch sbatch_script.sh ${ll} ${emb_dim} ${i} ${embd_constraint} ${label_count} ${MODELS_PER_JOB} 3586106167
                i=$((i+MODELS_PER_JOB))
                sbatch sbatch_script.sh ${ll} ${emb_dim} ${i} ${embd_constraint} ${label_count} ${MODELS_PER_JOB} 1325645050
                i=$((i+MODELS_PER_JOB))
                sbatch sbatch_script.sh ${ll} ${emb_dim} ${i} ${embd_constraint} ${label_count} ${MODELS_PER_JOB} 2564231920
                i=$((i+MODELS_PER_JOB))
              done


              if [ $emb_dim -eq 8 ]; then

                  for embd_constraint in 'gauss_moment3' 'gauss_moment4'
                  do

                    sbatch sbatch_script.sh ${ll} ${emb_dim} ${i} ${embd_constraint} ${label_count} ${MODELS_PER_JOB} 3474173998
                    i=$((i+MODELS_PER_JOB))
                    sbatch sbatch_script.sh ${ll} ${emb_dim} ${i} ${embd_constraint} ${label_count} ${MODELS_PER_JOB} 273230791
                    i=$((i+MODELS_PER_JOB))
                    sbatch sbatch_script.sh ${ll} ${emb_dim} ${i} ${embd_constraint} ${label_count} ${MODELS_PER_JOB} 3586106167
                    i=$((i+MODELS_PER_JOB))
                    sbatch sbatch_script.sh ${ll} ${emb_dim} ${i} ${embd_constraint} ${label_count} ${MODELS_PER_JOB} 1325645050
                    i=$((i+MODELS_PER_JOB))
                    sbatch sbatch_script.sh ${ll} ${emb_dim} ${i} ${embd_constraint} ${label_count} ${MODELS_PER_JOB} 2564231920
                    i=$((i+MODELS_PER_JOB))
                  done
              fi
        done
      done
    done
done

