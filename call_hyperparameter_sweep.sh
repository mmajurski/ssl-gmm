conda activate gmm

for i in {0..1000}
do
  python hyperparameter_optimizer.py
done