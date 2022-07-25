# Install anaconda3
# https://www.anaconda.com/distribution/
# conda config --set auto_activate_base false

# create a virtual environment to stuff all these packages into
conda create -n gmm python=3.8 -y

# activate the virtual environment
conda activate gmm

# install pytorch (best done through conda to handle cuda dependencies)
conda install pytorch=1.11 torchvision=0.12 cudatoolkit=11.3 -c pytorch

conda install pandas matplotlib

