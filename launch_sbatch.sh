#!/bin/bash


# find starting output directory
MODEL_DIR="./models-emb-const"
A=($MODEL_DIR/id-*)
HIGHEST_DIR="${A[-1]##*/}"
HIGHEST_VAL=$(echo $HIGHEST_DIR | tr -dc '0-9')
TRIM_HIGHEST=$(echo $HIGHEST_VAL | sed 's/^0*//')
TRIM_HIGHEST=1000
i=$((TRIM_HIGHEST+1))








MODELS_PER_JOB=1

for mn in 0
do

  for emb_dim in 0
  do
    for ood_p in 0.0
    do
    for label_count in 40 # 1, 4, and 25 per class
    do

          # sbatch sbatch_script.sh 'fc' ${emb_dim} ${i} 'none' ${trainer} ${label_count} ${MODELS_PER_JOB} ${ood_p} 0 0
          # i=$((i+MODELS_PER_JOB))

          for ll in "kmeans" "aa_gmm" "aa_gmm_d1"
          do
              for embd_constraint in 'l2' # 'none' 'mean_covar'
              do

                sbatch sbatch_script.sh ${ll} ${emb_dim} ${i} ${embd_constraint} ${label_count} ${MODELS_PER_JOB} ${ood_p} 0 0
                i=$((i+MODELS_PER_JOB))

                sbatch sbatch_script.sh ${ll} ${emb_dim} ${i} ${embd_constraint} ${label_count} ${MODELS_PER_JOB} ${ood_p} 0 1
                i=$((i+MODELS_PER_JOB))

                sbatch sbatch_script.sh ${ll} ${emb_dim} ${i} ${embd_constraint} ${label_count} ${MODELS_PER_JOB} ${ood_p} 1 0
                i=$((i+MODELS_PER_JOB))

                # already done in bulk
                #sbatch sbatch_script.sh ${ll} ${emb_dim} ${i} ${embd_constraint} ${label_count} ${MODELS_PER_JOB} ${ood_p} 1 1
                #i=$((i+MODELS_PER_JOB))
              done
        done
      done
    done
  done
done




# for mn in 0
# do

#   for emb_dim in 0 # 32
#   do
#     for ood_p in 0.2 0.5
#     do
#     for label_count in 40 # 1, 4, and 25 per class
#     do
#           # trainer='supervised'
#           # embd_constraint='none'
#           # sbatch sbatch_script.sh 'fc' ${learning_rate} ${emb_dim} ${i} ${embd_constraint} ${trainer} ${label_count} ${MODELS_PER_JOB}
#           # i=$((i+MODELS_PER_JOB))

#           trainer='fixmatch'
#           embd_constraint='none'
#           sbatch sbatch_script.sh 'fc' ${learning_rate} ${emb_dim} ${i} ${embd_constraint} ${trainer} ${label_count} ${MODELS_PER_JOB} ${ood_p}
#           i=$((i+MODELS_PER_JOB))

#           for ll in "kmeans" "aa_gmm" "aa_gmm_d1"
#           do
#               for embd_constraint in 'none' 'l2' 'mean_covar'
#               do
#                 trainer='fixmatch'
#                 sbatch sbatch_script.sh ${ll} ${learning_rate} ${emb_dim} ${i} ${embd_constraint} ${trainer} ${label_count} ${MODELS_PER_JOB} ${ood_p}
#                 i=$((i+MODELS_PER_JOB))
#               done
#         done
#       done
#     done
#   done
# done


