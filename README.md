# Shallowspeed

A tiny implementation of distributed training for sequential deep learning models.
Implemented using plain Numpy & mpi4py.


## Setups
```bash
conda env create
pip install -e .
# M1 Macs: conda install "libblas=*=*accelerate"
python download_dataset.py
```