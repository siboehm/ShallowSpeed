import argparse
import time
from pathlib import Path

import numpy as np
from mpi4py import MPI

from minMLP.dataset import Dataset
from minMLP.layers import MLP
from minMLP.optimizer import SGD
from minMLP.pipe import DataParallelSchedule, InferenceSchedule, Worker
from minMLP.utils import assert_sync, get_model_hash


def compute_accuracy(model, worker, dataset):
    """
    This function does a forward pass of x, then checks if the indices
    of the maximum value in the output equals the indices in the label
    y. Then it sums over each prediction and calculates the accuracy.
    """
    model.eval()

    correct = 0
    total = 0
    for batch_id in range(dataset.get_num_batches()):
        schedule = InferenceSchedule(
            num_micro_batches=1,
            num_stages=worker.pipeline_depth,
            stage_id=worker.stage_id,
        )
        worker.execute(schedule, batch_id)

        if worker.stage_id == worker.pipeline_depth - 1:
            pred = np.argmax(worker.output_buffers[0], axis=-1)
            target = np.argmax(dataset.load_micro_batch_target(batch_id, 0), axis=-1)
            correct += np.sum(pred == target)
            total += pred.shape[0]

    model.train()
    if worker.stage_id == worker.pipeline_depth - 1:
        return correct / total


EPOCHS = 20
# We use a big batch size, to make training more amenable to parallelization
GLOBAL_BATCH_SIZE = 128
N_MUBATCHES = 4


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dp",
        type=int,
        default=1,
        help="Degree of data parallelism (=number of full model replicas)",
    )
    parser.add_argument("--pp", type=int, default=1, help="Number of pipeline stages")
    parser.add_argument(
        "--schedule", type=str, choices=["pipedream", "naive"], default="naive"
    )
    args = parser.parse_args()
    DP_tile_factor = args.dp
    PP_tile_factor = args.pp

    assert DP_tile_factor >= 1 and PP_tile_factor >= 1
    assert DP_tile_factor * PP_tile_factor == MPI.COMM_WORLD.size, (
        f"Number of started workers is {MPI.COMM_WORLD.size}, "
        f"but should be {DP_tile_factor * PP_tile_factor} (DP * PP)"
    )
    assert (
        GLOBAL_BATCH_SIZE % DP_tile_factor == 0
    ), "Batch size must be properly divisible by DP"

    # create MPI communicators for data parallel AllReduce & pipeline parallel send & recv
    # if the `color=` parameter is the same, then those two workers end up in the same communicator
    dp_comm = MPI.COMM_WORLD.Split(color=MPI.COMM_WORLD.Get_rank() % PP_tile_factor)
    pp_comm = MPI.COMM_WORLD.Split(color=MPI.COMM_WORLD.Get_rank() // PP_tile_factor)
    # sanity check
    assert dp_comm.Get_size() == DP_tile_factor and pp_comm.Get_size() == PP_tile_factor

    # Set up the local model.
    # Layer_sizes is the total model size, which we split into PP-many stages
    layer_sizes = [784, 128, 127, 126, 125, 124, 123, 10]
    model = MLP(
        layer_sizes,
        stage_idx=pp_comm.rank,
        n_stages=PP_tile_factor,
        batch_size=GLOBAL_BATCH_SIZE,
    )
    model.train()

    optimizer = SGD(model.parameters(), lr=0.006)

    # Each DP-worker gets a slice of the global batch-size
    # TODO not every worker needs the dataset
    save_dir = Path("data/mnist_784/")
    local_batch_size = GLOBAL_BATCH_SIZE // DP_tile_factor
    dataset = Dataset(
        save_dir,
        global_batch_size=GLOBAL_BATCH_SIZE,
        mubatch_size=local_batch_size // N_MUBATCHES,
        validation=False,
    )
    dataset.load(dp_comm.Get_rank(), dp_comm.Get_size())
    worker = Worker(dp_comm, pp_comm, model, dataset, optimizer)

    val_dataset = Dataset(
        save_dir,
        global_batch_size=GLOBAL_BATCH_SIZE,
        mubatch_size=GLOBAL_BATCH_SIZE,
        validation=True,
    )
    val_dataset.load(DP_rank=0, DP_size=1)
    val_worker = Worker(None, pp_comm, model, val_dataset, None)

    start_time = time.time()
    for iteration in range(EPOCHS):
        accuracy = compute_accuracy(model, val_worker, val_dataset)
        if accuracy:
            print(
                f"Epoch: {iteration}, Time Spent: {time.time() - start_time:.2f}s, Accuracy: {accuracy * 100:.2f}%",
            )

        for batch_id in range(0, dataset.get_num_batches()):
            schedule = DataParallelSchedule(
                num_micro_batches=N_MUBATCHES,
                num_stages=PP_tile_factor,
                stage_id=pp_comm.rank,
            )
            # do the actual work
            worker.execute(schedule, batch_id)

    accuracy = compute_accuracy(model, val_worker, val_dataset)
    if accuracy is not None:
        print(
            f"Epoch: {EPOCHS}, Time Spent: {time.time() - start_time:.2f}s, Accuracy: {accuracy * 100:.2f}%",
        )

    # Sanity check: Make sure data parallel replicas have the same model weights
    assert_sync(dp_comm, get_model_hash(model))
