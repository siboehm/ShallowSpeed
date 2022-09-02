from minMLP.base import Parameter


class SGD:
    def __init__(self, parameters: list[Parameter], lr: float):
        # Boring stateless optimizer is boring
        self._params = parameters
        self._lr = lr

    def step(self):
        for param in self._params:
            if param.requires_grad:
                param.data -= self._lr * param.grad
