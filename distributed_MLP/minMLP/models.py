from mpi4py import MPI

from minMLP.base import Module
from minMLP.layers import NonLinearLayer, Linear, Softmax


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

    def train(self, mode=True):
        self._training = mode
        for l in self.layers:
            l.train(mode)

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
