# double-descent-friedman

## Setting up the Conda env

### Create environment with Python 3.11
```bash
conda create -n double_descent_env python=3.11 -y
```

### Activate environment
```bash
conda activate double_descent_env
```

### Install packages
```bash
conda install numpy scipy scikit-learn matplotlib pandas -y
```

### Using PyTorch
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

### For nicer plots
```bash
conda install seaborn -y
```

### Export environment
```bash
conda env export > environment.yml
```

## (Starting Point) To recreate the environment using the environment.yml
```bash
conda env create -f environment.yml
conda activate double_descent_env
```

To install ipykernel in your environment:
```bash
conda install ipykernel -y
```

Add the environment to Jupyter:
```bash
python -m ipykernel install --user --name=double_descent_env --display-name "Python (double_descent_env)"
```
--name is the internal name of the kernel

--display-name is what you will see in the Jupyter notebook

Launch:
```bash
jupyter notebook
```

Verify Packages:
```python
import sys
print(sys.executable)
import torch, sklearn, numpy
print(torch.__version__, sklearn.__version__)
```

After the conda environment has been activated, please run the following:
```bash
pip3 install "numpy<2"
pip3 install tqdm
pip3 install tqdm-joblib
pip3 intall --upgrade gmpy2
```
