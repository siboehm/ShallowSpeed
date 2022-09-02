from mpi4py import MPI

from minMLP.base import Module
from minMLP.layers import NonLinearLayer, Linear, Softmax


class Sequential(Module):
    def __init__(self, layers: list[Module]):
        super().__init__()
        self.layers = layers
        self._grad_hooks = {}
        self._post_grad_hooks = {}

    def forward(self, inputs):
        result = inputs
        for layer in self.layers:
            result = layer(result)
        return result

    def register_grad_hook(self, id, hook):
        """
        Register a hook to be run when the gradient for a parameter has been calculated
        """
        assert id not in self._grad_hooks
        self._grad_hooks[id] = hook

    def unregister_grad_hook(self, id):
        assert id in self._grad_hooks
        del self._grad_hooks[id]

    def register_post_grad_hook(self, id, hook):
        """
        Register a hook to be run before returning from the backwards()-function
        """
        assert id not in self._post_grad_hooks
        self._post_grad_hooks[id] = hook

    def unregister_post_grad_hook(self, id):
        assert id in self._post_grad_hooks
        del self._post_grad_hooks[id]

    def backward(self, dout):
        result = dout
        for layer in reversed(self.layers):
            result = layer.backward(result)

            for hook in self._grad_hooks.values():
                for param in layer.parameters():
                    hook(param)

        for hook in self._post_grad_hooks.values():
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
