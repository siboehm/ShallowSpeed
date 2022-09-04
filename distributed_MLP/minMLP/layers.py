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

    def __repr__(self):
        return "ReLU()"


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

    def __repr__(self):
        return "Softmax()"


class Linear(Module):
    def __init__(self, in_dims, out_dims, activation="relu"):
        super().__init__()
        assert activation is None or activation == "relu"

        # we want to get the same initial weights, no matter
        # if the model is distributed across workers or not
        rs = RandomState(MT19937(SeedSequence(in_dims + out_dims * 1337)))

        self.activation = ReLU() if activation == "relu" else None
        self._params["W"] = Parameter(
            rs.normal(0.0, 1.0, (out_dims, in_dims)) / np.sqrt(in_dims)
        )
        self._params["b"] = Parameter(np.zeros((1, out_dims)))

    def forward(self, input):
        if self._training:
            self._cache["input"] = input
        result = linear(input, self._params["W"].data, self._params["b"].data)

        if self.activation:
            return self.activation(result)
        return result

    def backward(self, dout):
        assert self._training

        if self.activation:
            dout = self.activation.backward(dout)

        dout, dW, db = linear_grad(dout, self._cache["input"], self._params["W"].data)

        # accumulate gradients
        self._params["W"].grad += dW
        self._params["b"].grad += db

        del self._cache["input"]
        return dout

    def __repr__(self):
        return f"Linear({self._params['W'].data.shape[1]}->{self._params['W'].data.shape[0]}, act: {self.activation})"


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

    def __repr__(self):
        return f"MSELoss()"
