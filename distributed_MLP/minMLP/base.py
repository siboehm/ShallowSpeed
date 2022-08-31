from abc import ABC, abstractmethod
import numpy as np


class Parameter(ABC):
    """Encapsulates a numpy array and keeps track of its gradient."""
    def __init__(self, data: np.array, requires_grad: bool = True):
        self.data = data
        self.grad = np.zeros_like(data)
        self.requires_grad = requires_grad

    def __repr__(self):
        return f"Parameter(shape={self.data.shape}, requires_grad={self.requires_grad})"


class Module(ABC):
    """
    A module is a stateful object, encapsulating a function and keeping track
    of the trainable parameters. It also keeps track of cached activations.
    """
    def __init__(self):
        self._params = {}
        self._cache = {}
        self._training = True

    def __call__(self, *inputs):
        return self.forward(*inputs)

    @abstractmethod
    def forward(self, *inputs):
        raise NotImplementedError

    @abstractmethod
    def backward(self, *dout):
        raise NotImplementedError

    def train(self, mode=True):
        self._training = mode

    def eval(self):
        self._training = False

    def zero_grad(self):
        for param in self._params.values():
            param.grad = None

    def parameters(self):
        return self._params.values()

    def named_parameters(self):
        return self._params.items()

    def state_dict(self):
        return {
            name: param.data for name, param in self.named_parameters()
        }

    def load_state_dict(self, state_dict):
        for name, param in self.named_parameters():
            param.data = state_dict[name]
