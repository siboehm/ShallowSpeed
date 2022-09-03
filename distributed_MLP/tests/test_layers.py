from minMLP.layers import Softmax, Linear, NonLinearLayer
from minMLP.models import Sequential
from minMLP.functional import mse_loss_grad
import numpy as np


def test_MLP_basic():
    layer_sizes = [132, 40, 11, 9]
    layers = [
        NonLinearLayer(layer_sizes[i], layer_sizes[i + 1])
        for i in range(len(layer_sizes) - 2)
    ]
    layers.append(Linear(layer_sizes[-2], layer_sizes[-1]))
    layers.append(Softmax())
    dnn = Sequential(layers)
    assert len(dnn.parameters()) == 6
    x = np.ones((13, 132))

    dnn.eval()
    output = dnn(x)
    assert output.shape == (13, 9)
    assert np.allclose(output.sum(), 13.0)

    dnn.train()
    output = dnn(x)
    target = np.diag(np.ones(9))
    target = np.concatenate((target, target[:4]))
    assert target.shape == (13, 9)

    dout = dnn.backward(mse_loss_grad(output, target, 13))
    # TODO: Make sure the last layer doesn't return the gradients wrt the input
    assert dout.shape == (13, 132)
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
