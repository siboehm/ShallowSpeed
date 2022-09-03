# distributed MLPs

A POC framework for training sequential models, distributed across CPUs.
Implemented using plain numpy & mpi4py.


## Setups
```bash
conda env create
pip install -e .
# M1 Macs: conda install "libblas=*=*accelerate"
# download data
python scripts/download_MNIST.py
```