import time
from pathlib import Path

import numpy as np
import pandas as pd
from mpi4py import MPI

from minMLP.layers import NonLinearLayer, Linear, Softmax, MSELoss
from minMLP.models import Sequential
from minMLP.optimizer import SGD
from minMLP.pipe import DataParallelSchedule, Worker, Dataset
from minMLP.utils import rprint, get_model_hash, assert_sync


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


EPOCHS = 10
# We use a big batch size, to make training more amenable to parallelization
GLOBAL_BATCH_SIZE = 128
N_MUBATCHES = 1


if __name__ == "__main__":
    # data parallel training only in this script
    DP_tile_factor = MPI.COMM_WORLD.size
    PP_tile_factor = 1

    save_dir = Path("../data/mnist_784/")
    assert save_dir.is_dir(), "Download the dataset first!"

    x_val = pd.read_parquet(save_dir / "x_val.parquet")
    y_val = np.load(save_dir / "y_val.npy")

    assert DP_tile_factor * PP_tile_factor == MPI.COMM_WORLD.size

    # create MPI communicators for data parallel AllReduce & pipeline parallel send & recv
    # if the `color=` parameter is the same, then those two workers end up in the same communicator
    dp_comm = MPI.COMM_WORLD.Split(color=MPI.COMM_WORLD.Get_rank() % PP_tile_factor)
    pp_comm = MPI.COMM_WORLD.Split(color=MPI.COMM_WORLD.Get_rank() // PP_tile_factor)
    # sanity check
    assert dp_comm.Get_size() == DP_tile_factor and pp_comm.Get_size() == PP_tile_factor

    layer_sizes = [784, 256, 128, 128, 10]
    layers = [
        NonLinearLayer(layer_sizes[i], layer_sizes[i + 1])
        for i in range(len(layer_sizes) - 2)
    ]
    layers.append(Linear(layer_sizes[-2], layer_sizes[-1]))
    layers.append(Softmax())
    layers.append(MSELoss(batch_size=GLOBAL_BATCH_SIZE))

    model = Sequential(layers)
    model.train()

    optimizer = SGD(model.parameters(), lr=0.01)

    # Each DP-worker gets a slice of the global batch-size
    # TODO not every worker needs the dataset
    assert GLOBAL_BATCH_SIZE % DP_tile_factor == 0
    batch_size = GLOBAL_BATCH_SIZE // DP_tile_factor
    dataset = Dataset(save_dir, batch_size, batch_size // N_MUBATCHES)
    dataset.load(dp_comm.Get_rank(), dp_comm.Get_size())
    worker = Worker(dp_comm, pp_comm, model, dataset, optimizer)

    start_time = time.time()
    for iteration in range(EPOCHS):
        accuracy = compute_accuracy(model, x_val, y_val)
        rprint(
            "Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2:.2f}%".format(
                iteration, time.time() - start_time, accuracy * 100
            ),
        )
        for batch_id in range(0, dataset.get_num_batches()):
            schedule = DataParallelSchedule(
                num_micro_batches=N_MUBATCHES,
                num_stages=pp_comm.size,
                stage_id=pp_comm.rank,
            )
            # do the actual work
            worker.execute(schedule, batch_id)

    accuracy = compute_accuracy(model, x_val, y_val)
    rprint(
        "Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2:.2f}%".format(
            EPOCHS, time.time() - start_time, accuracy * 100
        ),
    )

    # Sanity check: Make sure processes have the same model weights
    assert_sync(dp_comm, get_model_hash(model))
