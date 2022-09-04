import numpy as np

from minMLP.functional import mse_loss_grad
from minMLP.layers import MLP, Linear, ReLU, Sequential, Softmax


def test_MLP_basic():
    layer_sizes = [132, 40, 11, 9]
    layers = [
        Linear(
            layer_sizes[i],
            layer_sizes[i + 1],
            activation="relu" if i < len(layer_sizes) - 2 else None,
        )
        for i in range(len(layer_sizes) - 1)
    ]
    layers.append(Softmax())
    dnn = Sequential(layers)
    assert len(dnn.parameters()) == 6
    x = np.ones((13, 132), dtype=np.float32)

    dnn.eval()
    output = dnn(x)
    assert output.shape == (13, 9)
    assert output.dtype == np.float32
    assert np.allclose(output.sum(), 13.0)

    dnn.train()
    output = dnn(x)
    target = np.diag(np.ones(9, dtype=np.float32))
    target = np.concatenate((target, target[:4]))
    assert target.shape == (13, 9)

    dout = dnn.backward(mse_loss_grad(output, target, 13))
    # TODO: Make sure the last layer doesn't return the gradients wrt the input
    assert dout.shape == (13, 132)
    assert dout.dtype == np.float32
    # check if parameters were updated
    for param in dnn.parameters():
        assert param.requires_grad
        assert np.abs(param.grad).sum() > 0
        assert param.grad.shape == param.data.shape

    dnn.zero_grad()
    for param in dnn.parameters():
        assert np.abs(param.grad).sum() == 0
        assert param.grad.shape == param.data.shape

    assert len(dnn.parameters()) == 6


def test_distributed_MLP_init():
    layer_sizes = [1, 22, 98, 14, 132, 40, 11, 9, 33]
    n_stages = 3
    batch_size = 13

    # first in pipeline
    dnn = MLP(layer_sizes, 0, n_stages, batch_size)
    assert len(dnn.parameters()) == 2 * 3
    assert len(dnn.layers) == 3
    assert all(isinstance(l.activation, ReLU) for l in dnn.layers)
    assert dnn.in_dim == 1 and dnn.out_dim == 14

    # last in pipeline
    dnn = MLP(layer_sizes, 2, n_stages, batch_size)
    assert len(dnn.parameters()) == 2 * 2
    assert len(dnn.layers) == 4
    assert isinstance(dnn.layers[0].activation, ReLU)
    assert dnn.layers[1].activation is None
    assert dnn.in_dim == 11 and dnn.out_dim == 33
