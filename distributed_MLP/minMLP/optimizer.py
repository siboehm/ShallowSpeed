class SGD:
    def __init__(self, parameters: list, lr: float):
        self._params = parameters
        self._lr = lr

    def step(self):
        for param in self._params:
            param.data -= self._lr * param.grad
