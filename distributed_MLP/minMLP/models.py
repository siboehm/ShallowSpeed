from mpi4py import MPI

from minMLP.base import Module
from minMLP.layers import NonLinearLayer, Linear, Softmax


class Sequential(Module):
    def __init__(self, layers: list[Module]):
        super().__init__()
        self.layers = layers

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
    def __init__(self, sizes: list[int]):
        assert len(sizes) >= 2
        self.layers = [
            NonLinearLayer(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 2)
        ]
        self.layers.append(Linear(sizes[-2], sizes[-1]))
        self.layers.append(Softmax())

        super().__init__(self.layers)


class DP_Sequential(Sequential):
    def __init__(self, layers: list[Module], comm=MPI.COMM_WORLD):
        super().__init__(layers)
        self.comm = comm

    def backward(self, dout):
        # rescale the gradient to account for the mean in the loss
        result = dout / self.comm.size
        futures = []

        for layer in reversed(self.layers):
            result = layer.backward(result)

            # iterate over Parameters for which we just calculated the gradient
            for param in layer.parameters():
                if param.requires_grad:
                    # start a non-blocking allReduce
                    futures.append(
                        self.comm.Iallreduce(
                            sendbuf=MPI.IN_PLACE, recvbuf=param.grad, op=MPI.SUM
                        )
                    )

        # after the backwards pass, wait for all communication to finish
        MPI.Request.Waitall(futures)

        # rescale the final gradients
        for param in self.parameters():
            if param.requires_grad:
                param.grad /= self.comm.size

        return result
