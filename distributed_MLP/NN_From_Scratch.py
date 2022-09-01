import time

import pandas as pd
import time
from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import torch
from mpi4py import MPI
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from minMLP.functional import mse_loss, mse_loss_grad
from minMLP.models import MLP
from minMLP.optimizer import SGD


def compute_accuracy(model, x_val, y_val):
    """
    This function does a forward pass of x, then checks if the indices
    of the maximum value in the output equals the indices in the label
    y. Then it sums over each prediction and calculates the accuracy.
    """
    x_val = x_val.to_numpy()

    model.eval()
    output = model.forward(x_val)
    model.train()

    pred = np.argmax(output, axis=-1)
    target = np.argmax(y_val, axis=-1)
    return np.mean(pred == target)


def download_dataset(save_dir):
    print("Downloading the dataset at", save_dir.resolve())
    x, y = fetch_openml("mnist_784", version=1, data_home="data_cache", return_X_y=True)

    x /= 255.0
    x -= x.mean()
    y = pd.get_dummies(y)

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.15, random_state=42
    )
    save_dir.mkdir()
    x_train.to_parquet(save_dir / "x_train.parquet")
    x_val.to_parquet(save_dir / "x_val.parquet")
    np.save(save_dir / "y_train.npy", y_train)
    np.save(save_dir / "y_val.npy", y_val)


EPOCHS = 20
BATCH_SIZE = 64

if __name__ == "__main__":
    # init MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    save_dir = Path("../data/mnist_784/")
    if not save_dir.is_dir():
        download_dataset(save_dir)

    x_train = pd.read_parquet(save_dir / "x_train.parquet").to_numpy()
    x_val = pd.read_parquet(save_dir / "x_val.parquet")
    y_train = np.load(save_dir / "y_train.npy")
    y_val = np.load(save_dir / "y_val.npy")

    dnn = MLP(sizes=[784, 100, 10])
    dnn.train()
    optimizer = SGD(dnn.parameters(), lr=0.03)

    start_time = time.time()
    dnn.train()
    for iteration in range(EPOCHS):
        accuracy = compute_accuracy(dnn, x_val, y_val)
        print(
            "Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2:.2f}%".format(
                iteration, time.time() - start_time, accuracy * 100
            )
        )
        for j in range(0, len(x_train), BATCH_SIZE):
            x = x_train[j : min(len(x_train), j + BATCH_SIZE), ...]
            y = y_train[j : min(len(x_train), j + BATCH_SIZE)]

            output = dnn.forward(x)
            loss = mse_loss(output, y)
            dout = mse_loss_grad(output, y)

            dnn.zero_grad()
            dnn.backward(dout)
            optimizer.step()

    accuracy = compute_accuracy(dnn, x_val, y_val)
    print(
        "Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2:.2f}%".format(
            EPOCHS, time.time() - start_time, accuracy * 100
        )
    )
