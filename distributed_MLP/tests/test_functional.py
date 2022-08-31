import numpy as np
from minMLP.functional import (
    relu,
    linear,
    softmax,
    relu_grad,
    linear_grad,
    softmax_grad,
)

EPS = 10e-6


def test_shapes():
    # relu
    x = np.empty((2, 3))
    y = relu(x)
    dinput = relu_grad(y, x)
    assert x.shape == dinput.shape

    # linear
    weight = np.empty((3, 10))
    bias = np.empty((3,))
    x = np.empty((20, 10))
    y = linear(x, weight, bias)
    assert y.shape == (20, 3)
    dinput, dweight, dbias = linear_grad(np.empty((20, 3)), x, weight, bias)
    assert dinput.shape == x.shape
    assert dweight.shape == weight.shape
    assert dbias.shape == bias.shape

    # softmax
    x = np.empty((20, 10))
    y = softmax(x)
    assert y.shape == (20, 10)
    dinput = softmax_grad(y, x)
    assert dinput.shape == x.shape


def test_relu_grad():
    x = np.array([[-1, -2, -3], [0.1, 5, 6]])
    finite_diff = (relu(x + EPS / 2) - relu(x - EPS / 2)) / EPS
    assert np.allclose(relu_grad(np.ones_like(x), x), finite_diff)


def I_ij(i, j, n, m):
    # 1 at position i,j zero otherwise
    result = np.zeros((n, m))
    result[i, j] += 1
    return result


def test_linear_grad():
    x = np.array([[-1, -2, -3]], dtype=float)
    W = np.array([[2, 3, -1], [1, 0, 4], [9, -9, 1], [1, -3, 5]], dtype=float)
    b = np.array([[1, -1, 1, 3]], dtype=float)
    grad_out = np.arange(4, dtype=float).reshape((1, 4))

    # calculating Jacobian for input using finite differences method
    jacobian_i_fd = np.zeros((3, 4), dtype=float)
    for i in range(3):
        for o in range(4):
            jacobian_i_fd[i, o] += (
                (
                    linear(x + EPS / 2 * I_ij(0, i, 1, 3), W, b)
                    - linear(x - EPS / 2 * I_ij(0, i, 1, 3), W, b)
                )
                / EPS
            )[0][o]
    jvp = jacobian_i_fd @ grad_out[0]
    real = linear_grad(grad_out, x, W, b)[0]
    assert np.allclose(jvp, real)

    # calculating Jacobian for weights using finite differences method
    jacobian_W_fd = np.zeros((4, 3, 4), dtype=float)
    for r in range(4):
        for c in range(3):
            for o in range(4):
                jacobian_W_fd[r, c, o] += (
                    (
                        linear(x, W + EPS / 2 * I_ij(r, c, 4, 3), b)
                        - linear(x, W - EPS / 2 * I_ij(r, c, 4, 3), b)
                    )
                    / EPS
                )[0][o]
    jvp = jacobian_W_fd @ grad_out[0]
    real = linear_grad(grad_out, x, W, b)[1]
    assert np.allclose(jvp, real)

    # calculating Jacobian for bias using finite differences method
    jacobian_b_fd = np.zeros((4, 4), dtype=float)
    for b in range(4):
        for o in range(4):
            jacobian_b_fd[b, o] += (
                (
                    linear(x, W, b + EPS / 2 * I_ij(0, b, 1, 4))
                    - linear(x, W, b - EPS / 2 * I_ij(0, b, 1, 4))
                )
                / EPS
            )[0][o]
    jvp = jacobian_b_fd @ grad_out[0]
    real = linear_grad(grad_out, x, W, b)[2]
    assert np.allclose(jvp, real)


def test_softmax():
    x = np.array([[1, 2, 3], [4, 5, 6]])
    y = softmax(x)
    assert np.allclose(y.sum(axis=1), np.ones(y.shape[0]))
    # softmax is invariant to shifts
    assert np.allclose(softmax(x), softmax(x - 6))


def test_softmax_grad():
    x = np.array([[-1, -2, -3]], dtype=float)
    grad_out = np.array([[1, 9, 11]], dtype=float)

    # calculating Jacobian for input using finite differences method
    jacobian_i_fd = np.zeros((3, 3), dtype=float)
    for i_i in range(3):
        for o_i in range(3):
            jacobian_i_fd[i_i, o_i] += (
                (
                    softmax(x + EPS / 2 * I_ij(0, i_i, 1, 3))
                    - softmax(x - EPS / 2 * I_ij(0, i_i, 1, 3))
                )
                / EPS
            )[0][o_i]

    jvp = jacobian_i_fd @ grad_out[0]
    real = softmax_grad(grad_out, x)
    assert np.allclose(jvp, real)
