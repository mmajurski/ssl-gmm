#!/bin/bash


# find starting output directory
MODEL_DIR="./models"
A=($MODEL_DIR/id-*)
HIGHEST_DIR="${A[-1]##*/}"
HIGHEST_VAL=$(echo $HIGHEST_DIR | tr -dc '0-9')
TRIM_HIGHEST=$(echo $HIGHEST_VAL | sed 's/^0*//')
i=$((TRIM_HIGHEST+1))



MODELS_PER_JOB=1

# sbatch sbatch_script.sh "aa_gmm" 32 ${i} "none" 40 1 0 1
# i=$((i+MODELS_PER_JOB))
# sbatch sbatch_script.sh "aa_gmm" 32 ${i} "none" 40 1 0 1
# i=$((i+MODELS_PER_JOB))
# sbatch sbatch_script.sh "aa_gmm" 32 ${i} "none" 40 1 0 1
# i=$((i+MODELS_PER_JOB))
# sbatch sbatch_script.sh "aa_gmm" 32 ${i} "none" 40 1 0 1
# i=$((i+MODELS_PER_JOB))

# sbatch sbatch_script.sh "aa_gmm" 32 ${i} "l2" 250 1 0 1
# i=$((i+MODELS_PER_JOB))
# sbatch sbatch_script.sh "aa_gmm" 32 ${i} "l2" 250 1 0 1
# i=$((i+MODELS_PER_JOB))

# sbatch sbatch_script.sh "aa_gmm" 32 ${i} "none" 250 1 0 1
# i=$((i+MODELS_PER_JOB))
# sbatch sbatch_script.sh "aa_gmm" 32 ${i} "none" 250 1 0 1
# i=$((i+MODELS_PER_JOB))

# sbatch sbatch_script.sh "kmeans" 32 ${i} "l2" 250 1 0 1
# i=$((i+MODELS_PER_JOB))
# sbatch sbatch_script.sh "kmeans" 32 ${i} "l2" 250 1 0 1
# i=$((i+MODELS_PER_JOB))

# sbatch sbatch_script.sh "kmeans" 32 ${i} "none" 250 1 0 1
# i=$((i+MODELS_PER_JOB))
# sbatch sbatch_script.sh "kmeans" 32 ${i} "none" 250 1 0 1
# i=$((i+MODELS_PER_JOB))

# sbatch sbatch_script.sh "aa_gmm" 0 ${i} "none" 40 1 0 0
# i=$((i+MODELS_PER_JOB))
# sbatch sbatch_script.sh "aa_gmm" 0 ${i} "none" 40 1 0 0
# i=$((i+MODELS_PER_JOB))

# sbatch sbatch_script.sh "aa_gmm_d1" 32 ${i} "none" 250 1 0 1
# i=$((i+MODELS_PER_JOB))
# sbatch sbatch_script.sh "aa_gmm_d1" 32 ${i} "none" 250 1 0 1
# i=$((i+MODELS_PER_JOB))

# sbatch sbatch_script.sh "aa_gmm_d1" 32 ${i} "l2" 250 1 0 1
# i=$((i+MODELS_PER_JOB))
# sbatch sbatch_script.sh "aa_gmm_d1" 32 ${i} "l2" 250 1 0 1
# i=$((i+MODELS_PER_JOB))

# sbatch sbatch_script.sh "aa_gmm_d1" 32 ${i} "none" 40 1 0 1
# i=$((i+MODELS_PER_JOB))
# sbatch sbatch_script.sh "aa_gmm_d1" 32 ${i} "none" 40 1 0 1
# i=$((i+MODELS_PER_JOB))
# sbatch sbatch_script.sh "aa_gmm_d1" 32 ${i} "none" 40 1 0 1
# i=$((i+MODELS_PER_JOB))
# sbatch sbatch_script.sh "aa_gmm_d1" 32 ${i} "none" 40 1 0 1
# i=$((i+MODELS_PER_JOB))

# sbatch sbatch_script.sh "aa_gmm" 32 ${i} "l2" 40 1 0 1
# i=$((i+MODELS_PER_JOB))
# sbatch sbatch_script.sh "aa_gmm" 32 ${i} "l2" 40 1 0 1
# i=$((i+MODELS_PER_JOB))

# sbatch sbatch_script.sh "aa_gmm" 0 ${i} "l2" 250 1 0 0
# i=$((i+MODELS_PER_JOB))
# sbatch sbatch_script.sh "aa_gmm" 0 ${i} "l2" 250 1 0 0
# i=$((i+MODELS_PER_JOB))

# sbatch sbatch_script.sh "kmeans" 32 ${i} "none" 40 1 0 1
# i=$((i+MODELS_PER_JOB))
# sbatch sbatch_script.sh "kmeans" 32 ${i} "none" 40 1 0 1
# i=$((i+MODELS_PER_JOB))

# sbatch sbatch_script.sh "kmeans" 0 ${i} "none" 40 1 0 1
# i=$((i+MODELS_PER_JOB))
# sbatch sbatch_script.sh "kmeans" 0 ${i} "none" 40 1 0 1
# i=$((i+MODELS_PER_JOB))
# sbatch sbatch_script.sh "kmeans" 0 ${i} "none" 40 1 0 1
# i=$((i+MODELS_PER_JOB))
# sbatch sbatch_script.sh "kmeans" 0 ${i} "none" 40 1 0 1
# i=$((i+MODELS_PER_JOB))
# sbatch sbatch_script.sh "kmeans" 0 ${i} "none" 40 1 0 1
# i=$((i+MODELS_PER_JOB))

# sbatch sbatch_script.sh "aa_gmm" 32 ${i} "l2" 40 1 0 0
# i=$((i+MODELS_PER_JOB))
# sbatch sbatch_script.sh "aa_gmm" 32 ${i} "l2" 40 1 0 0
# i=$((i+MODELS_PER_JOB))











for mn in 0 1
do

  for emb_dim in 0 32
  do
    for ood_p in 0
    do
    for label_count in 40 # 1, 4, and 25 per class
    do

          # sbatch sbatch_script.sh 'fc' ${emb_dim} ${i} 'none' ${label_count} ${MODELS_PER_JOB} ${ood_p}
          # i=$((i+MODELS_PER_JOB))

          for ll in "kmeans" "aa_gmm" "aa_gmm_d1"
          do
              for embd_constraint in 'none' 'l2' # 'mean_covar'
              do
                sbatch sbatch_script.sh ${ll} ${emb_dim} ${i} ${embd_constraint} ${label_count} ${MODELS_PER_JOB} ${ood_p}
                i=$((i+MODELS_PER_JOB))
              done
        done
      done
    done
  done
done

