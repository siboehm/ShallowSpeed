import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence

from minMLP.base import Module, Parameter
from minMLP.functional import (
    linear,
    linear_grad,
    mse_loss_grad,
    relu,
    relu_grad,
    softmax,
    softmax_grad,
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
            rs.normal(0.0, 1.0, (out_dims, in_dims)).astype(np.float32)
            / np.sqrt(in_dims)
        )
        self._params["b"] = Parameter(np.zeros((1, out_dims), dtype=np.float32))

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


class Sequential(Module):
    def __init__(self, layers: list[Module]):
        super().__init__()
        self.layers = layers
        self._grad_hooks = []
        self._post_grad_hooks = []

    def forward(self, inputs):
        result = inputs
        for layer in self.layers:
            result = layer(result)
        return result

    def register_grad_hook(self, hook):
        """
        Register a hook to be run when the gradient for a parameter has been calculated
        """
        assert id not in self._grad_hooks
        self._grad_hooks.append(hook)

    def reset_grad_hooks(self):
        self._grad_hooks = []

    def register_post_grad_hook(self, hook):
        """
        Register a hook to be run before returning from the backwards()-function
        """
        self._post_grad_hooks.append(hook)

    def reset_post_grad_hooks(self):
        self._post_grad_hooks = []

    def backward(self, dout):
        result = dout
        for layer in reversed(self.layers):
            result = layer.backward(result)

            for hook in self._grad_hooks:
                for param in layer.parameters():
                    hook(param)

        for hook in self._post_grad_hooks:
            hook(self.parameters())

        return result

    def train(self):
        self._training = True
        for l in self.layers:
            l.train()

    def eval(self):
        self._training = False
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


class MLP(Sequential):
    def __init__(self, sizes: list[int], stage_idx, n_stages, batch_size):
        """
        :param batch_size: The total batch size. This is necessary for rescaling the
            loss while operating on DP-slices & μBatches
        """
        assert len(sizes) % n_stages == 0
        stage_size = len(sizes) // n_stages

        # construct & init local layers
        is_last_stage = stage_idx == n_stages - 1
        local_sizes = sizes[
            stage_idx
            * stage_size : min(len(sizes), stage_size * stage_idx + stage_size + 1)
        ]
        layers = [
            Linear(
                local_sizes[i],
                local_sizes[i + 1],
                activation=None
                if i == len(local_sizes) - 2 and is_last_stage
                else "relu",
            )
            for i in range(len(local_sizes) - 1)
        ]
        if is_last_stage:
            layers.append(Softmax())
            layers.append(MSELoss(batch_size=batch_size))
        super().__init__(layers)

        print(layers)

        self.in_dim = local_sizes[0]
        # softmax & losses don't change output dimensions
        self.out_dim = local_sizes[-1]
