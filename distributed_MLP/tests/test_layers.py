from minMLP.layers import MLP
from minMLP.functional import mse_loss_grad
import numpy as np


def test_MLP_basic():
    dnn = MLP(sizes=[132, 11, 9])
    assert len(dnn.parameters()) == 4
    x = np.ones((13, 132))

    dnn.eval()
    output = dnn(x)
    assert output.shape == (13, 9)

    dnn.train()
    output = dnn(x)
    assert np.allclose(output.sum(), 13.0)

    target = np.diag(np.ones(9))
    target = np.concatenate((target, target[:4]))
    assert target.shape == (13, 9)
    dout = dnn.backward(mse_loss_grad(output, target))
    for param in dnn.parameters():
        assert param.requires_grad
        assert np.abs(param.grad).sum() > 0
        assert param.grad.shape == param.data.shape

    dnn.zero_grad()
    for param in dnn.parameters():
        assert np.abs(param.grad).sum() == 0
        assert param.grad.shape == param.data.shape
