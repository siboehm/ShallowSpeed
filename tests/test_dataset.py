from pathlib import Path

import numpy as np
import pandas as pd

from minMLP.dataset import Dataset


def test_dataset():
    save_path = Path("data/mnist_784")
    dataset = Dataset(save_path, 128, 8)
    input_X = pd.read_parquet(save_path / f"x_train.parquet").to_numpy()

    num_sample = 59500
    num_sample_no_tile_quantization = num_sample - (num_sample % 128)
    dataset.load(DP_rank=1, DP_size=4)
    assert len(dataset) == num_sample_no_tile_quantization // 4
    assert dataset.load_micro_batch_input(0, 0).dtype == np.float32
