import numpy as np
from minMLP.base import Parameter, Module
from minMLP.functional import (
    relu,
    relu_grad,
    linear,
    linear_grad,
    softmax_grad,
    softmax,
)


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
        self._params["W"] = Parameter(np.random.standard_normal((out_dims, in_dims)))
        self._params["b"] = Parameter(np.zeros((1, out_dims)))

    def forward(self, input):
        if self._training:
            self._cache["input"] = input
        return linear(input, self._params["W"].data, self._params["b"].data)

    def backward(self, dout):
        assert self._training
        dout, dW, db = linear_grad(
            dout, self._cache["input"], self._params["W"].data, self._params["b"].data
        )

        # accumulate gradients
        self._params["W"].grad += dW
        self._params["b"].grad += db

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


class MLP(Module):
    def __init__(self, sizes: list):
        super().__init__()

        assert len(sizes) >= 2
        self.layers = [
            NonLinearLayer(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 2)
        ]
        self.layers.append(Linear(sizes[-2], sizes[-1]))
        self.layers.append(Softmax())

    def forward(self, inputs):
        result = inputs
        for layer in self.layers:
            result = layer(result)
        return result

    def backward(self, dout):
        result = dout
        for layer in reversed(self.layers):
            result = layer.backward(result)
        return result

    def train(self, mode=True):
        for l in self.layers:
            l.train(mode)

    def eval(self):
        for l in self.layers:
            l.eval()

    def zero_grad(self):
        for l in self.layers:
            l.zero_grad()

    def parameters(self):
        result = []
        for l in self.layers:
            result += l.parameters()
        return result
