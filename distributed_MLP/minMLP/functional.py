import numpy as np


def relu(input):
    return input.clip(min=0.0)


def relu_grad(grad_output, bitmask):
    return grad_output * bitmask


def linear(input, weight, bias):
    """
    y = x@A^T + b
    """
    return input @ weight.T + bias


def linear_grad(grad_output, input, weight, bias):
    return grad_output @ weight, grad_output.T @ input, grad_output.sum(axis=0)


def softmax(input):
    # logsumexp trick
    input_exp = np.exp(input - np.max(input))
    return input_exp / input_exp.sum(axis=1, keepdims=True)


def softmax_grad(grad_output, input):
    # ideally we would cache the output instead of the input during FW,
    # avoiding the recomputation
    output = softmax(input)
    new_grad = output * grad_output
    return new_grad - output * new_grad.sum(axis=-1, keepdims=True)

