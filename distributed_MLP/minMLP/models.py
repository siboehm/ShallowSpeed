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
        self._grad_hooks.append(hook)

    def register_post_grad_hook(self, hook):
        """
        Register a hook to be run before returning from the backwards()-function
        """
        self._post_grad_hooks.append(hook)

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


class MLP(Sequential):
    def __init__(self, sizes: list[int], comm=None):
        assert len(sizes) >= 2
        self.layers = [
            NonLinearLayer(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 2)
        ]
        self.layers.append(Linear(sizes[-2], sizes[-1]))
        self.layers.append(Softmax())

        super().__init__(self.layers)

        self.comm = comm
        # setup hooks for data-parallel backprop
        if self.comm is not None:
            # start a non-blocking AllReduce for the parameters for which we just
            # calculated the final gradient.
            # This interleaves communication of this layer's gradients with
            # computation of the next layers gradients
            def allreduce_gradient(param):
                if param.requires_grad:
                    param._request = self.comm.Iallreduce(
                        sendbuf=MPI.IN_PLACE, recvbuf=param.grad, op=MPI.SUM
                    )

            self.register_grad_hook(allreduce_gradient)

            # after the full backwards pass we wait for all communication to finish
            # only then can we be certain that the gradients are the same on all processes
            def wait_for_comms(params):
                requests = [
                    param._request
                    for param in params
                    if param.requires_grad and param._request is not None
                ]
                MPI.Request.Waitall(requests)

            self.register_post_grad_hook(wait_for_comms)

    def backward(self, dout):
        # rescale the gradient to account for the average-ing in the loss
        if self.comm is not None:
            dout /= self.comm.size
        return super().backward(dout)
