import numpy as np
import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.layers import Dense, ReLU, Softmax


def test_dense_forward_shape():
    layer = Dense(4, 3)

    x = np.random.randn(5, 4)

    output = layer.forward(x)

    assert output.shape == (5, 3)


def test_dense_backward_shape():
    layer = Dense(4, 3)

    x = np.random.randn(5, 4)

    layer.forward(x)

    grad = np.random.randn(5, 3)

    dx = layer.backward(grad)

    assert dx.shape == (5, 4)
    assert layer.d_weights.shape == (4, 3)
    assert layer.d_biases.shape == (1, 3)


def test_relu_forward():
    relu = ReLU()

    x = np.array([[-1, 2, -3, 4]])

    output = relu.forward(x)

    expected = np.array([[0, 2, 0, 4]])

    assert np.array_equal(output, expected)


def test_relu_backward():
    relu = ReLU()

    x = np.array([[-1, 2, -3, 4]])

    relu.forward(x)

    grad = np.ones_like(x)

    output = relu.backward(grad)

    expected = np.array([[0, 1, 0, 1]])

    assert np.array_equal(output, expected)


def test_softmax_probabilities_sum_to_one():
    softmax = Softmax()

    x = np.random.randn(10, 5)

    probs = softmax.forward(x)

    sums = np.sum(probs, axis=1)

    assert np.allclose(sums, 1.0)