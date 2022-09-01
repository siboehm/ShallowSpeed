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
        assert comm.size > 1

    def backward(self, dout):
        # rescale the gradient to account for the average-ing in the loss
        result = dout / self.comm.size
        requests = []

        for layer in reversed(self.layers):
            result = layer.backward(result)
            # start a non-blocking AllReduce for the parameters for which we just
            # calculated the final gradient.
            # This interleaves communication of this layer's gradients with
            # computation of the next layers gradients
            for param in layer.parameters():
                if param.requires_grad:
                    requests.append(
                        self.comm.Iallreduce(
                            sendbuf=MPI.IN_PLACE, recvbuf=param.grad, op=MPI.SUM
                        )
                    )

        # after the full backwards pass we wait for all communication to finish
        # only then can we be certain that the gradients are the same on all processes
        MPI.Request.Waitall(requests)

        return result


class Distributed_MLP(DP_Sequential):
    def __init__(self, sizes: list[int], comm=MPI.COMM_WORLD):
        assert len(sizes) >= 2
        self.layers = [
            NonLinearLayer(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 2)
        ]
        self.layers.append(Linear(sizes[-2], sizes[-1]))
        self.layers.append(Softmax())

        super().__init__(self.layers, comm=comm)
