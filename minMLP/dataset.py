import numpy as np
import pandas as pd


class Dataset:
    input_X = None
    target_y = None

    def __init__(
        self,
        save_dir,
        global_batch_size,
        mubatch_size,
        validation=False,
    ):
        assert save_dir.is_dir(), "Download the dataset first!"
        self.save_dir = save_dir
        self.global_batch_size = global_batch_size
        self.local_batch_size = None
        self.mubatch_size = mubatch_size
        self._val = validation

    def load(self, DP_rank, DP_size):
        assert DP_rank < DP_size
        assert self.global_batch_size % DP_size == 0
        assert (
            self.global_batch_size // DP_size
        ) % self.mubatch_size == 0, "μBatchsize must divide batchsize!"
        self.local_batch_size = self.global_batch_size // DP_size

        # each process loads the whole dataset
        # this is inefficient for large datasets, but fine for tiny MNIST
        suffix = "val" if self._val else "train"
        input_X = pd.read_parquet(self.save_dir / f"x_{suffix}.parquet").to_numpy(
            dtype=np.float32
        )
        target_y = np.load(self.save_dir / f"y_{suffix}.npy").astype(np.float32)
        assert len(input_X) == len(target_y)

        # drop last few samples such that each batch is exactly `global_batch_size` long
        # this is important to ensure equivalence when changing the number of μBatches
        full_tiles_length = len(input_X) - (len(input_X) % self.global_batch_size)

        # each DP process selects its subset of the datasets by a `rank`-offset and `size`-strides
        # the copy() is super important, else the array is not continuous in memory
        # which results in horrible matmul performance
        self.input_X = input_X[DP_rank:full_tiles_length:DP_size].copy()
        self.target_y = target_y[DP_rank:full_tiles_length:DP_size].copy()

        assert len(self.input_X) % self.mubatch_size == 0
        assert len(self.input_X) % self.local_batch_size == 0

    def __len__(self):
        return len(self.input_X)

    def load_micro_batch_input(self, batch_id, mubatch_id):
        assert batch_id < self.get_num_batches()
        assert mubatch_id < self.get_num_mubatches()
        start_idx = batch_id * self.local_batch_size + mubatch_id * self.mubatch_size
        end_idx = start_idx + self.mubatch_size
        assert end_idx <= len(self.input_X)
        return self.input_X[start_idx:end_idx]

    def load_micro_batch_target(self, batch_id, mubatch_id):
        assert batch_id < self.get_num_batches()
        assert mubatch_id < self.get_num_mubatches()
        start_idx = batch_id * self.local_batch_size + mubatch_id * self.mubatch_size
        end_idx = start_idx + self.mubatch_size
        assert end_idx <= len(self.input_X)
        return self.target_y[start_idx:end_idx]

    def get_num_batches(self):
        return len(self) // self.local_batch_size

    def get_num_mubatches(self):
        return self.local_batch_size // self.mubatch_size
