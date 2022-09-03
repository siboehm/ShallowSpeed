import numpy as np
from minMLP.base import Module, Parameter
from minMLP.functional import (
    linear,
    linear_grad,
    mse_loss,
    mse_loss_grad,
    relu,
    relu_grad,
    softmax,
    softmax_grad,
)
from numpy.random import MT19937, RandomState, SeedSequence

rs = RandomState(MT19937(SeedSequence(123456789)))


class ReLU(Module):
    def forward(self, input):
        if self._training:
            self._cache["bitmask"] = input > 0
        return relu(input)

    def backward(self, dout):
        assert self._training
        dout = relu_grad(dout, self._cache["bitmask"])
        del self._cache["bitmask"]
        return dout


class Softmax(Module):
    def forward(self, inputs):
        if self._training:
            self._cache["input"] = inputs
        return softmax(inputs)

    def backward(self, dout):
        assert self._training
        dout = softmax_grad(dout, self._cache["input"])
        del self._cache["input"]
        return dout


class Linear(Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        # should probably be scaled by 1 / sqrt(in_dims)
        self._params["W"] = Parameter(rs.uniform(-0.1, 0.1, (out_dims, in_dims)))
        self._params["b"] = Parameter(np.zeros((1, out_dims)))

    def forward(self, input):
        if self._training:
            self._cache["input"] = input
        return linear(input, self._params["W"].data, self._params["b"].data)

    def backward(self, dout):
        assert self._training
        dout, dW, db = linear_grad(dout, self._cache["input"], self._params["W"].data)

        # accumulate gradients
        self._params["W"].grad += dW
        self._params["b"].grad += db

        del self._cache["input"]
        return dout


class MSELoss(Module):
    def __init__(self, batch_size: int):
        super().__init__()
        self.batch_size = batch_size

    # You don't need to calculate the loss to compute the gradient
    # so we just don't do it
    def forward(self, input: np.array):
        if self._training:
            self._cache["input"] = input
        return input

    def backward(self, target):
        assert self._training
        dout = mse_loss_grad(self._cache["input"], target, self.batch_size)
        del self._cache["input"]
        return dout


class NonLinearLayer(Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.linear = Linear(in_dims, out_dims)
        self.relu = ReLU()
        self._params = self.linear._params

    def forward(self, inputs):
        return self.relu(self.linear(inputs))

    def backward(self, dout):
        return self.linear.backward(self.relu.backward(dout))
