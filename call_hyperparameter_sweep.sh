conda activate gmm

#for i in {0..50}
#do
#  python hyperparameter_optimizer.py
#done


INDEX=0
TOTAL_MODEL_COUNT=50

# start the first 2 trains right away
python hyperparameter_optimizer.py &
INDEX=$((INDEX+1))
sleep 1
python hyperparameter_optimizer.py &
INDEX=$((INDEX+1))
sleep 1
python hyperparameter_optimizer.py &
INDEX=$((INDEX+1))
sleep 1

# loop over the remainder of the count to be generated, and when one of the two running jobs completes, launch a new one
for i in $(seq $TOTAL_MODEL_COUNT); do

  wait -n
  python hyperparameter_optimizer.py &
  INDEX=$((INDEX+1))
  sleep 1
done

# wait for all of the runs to complete before exiting
wait