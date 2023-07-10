# Install anaconda3
# https://www.anaconda.com/distribution/
# conda config --set auto_activate_base false

# create a virtual environment to stuff all these packages into
conda create -n gmm python=3.11 -y

# activate the virtual environment
conda activate gmm

# install pytorch (best done through conda to handle cuda dependencies)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

conda install pandas matplotlib scikit-learn jsonpickle psutil -y

#pip install optuna
