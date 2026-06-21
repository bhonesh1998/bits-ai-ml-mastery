import numpy as np
import pytest
from src.layers import Dense, ReLU

def test_dense_forward_shapes():
    # Ensures matrix multiplication dimensions align correctly
    layer = Dense(input_dim=4, output_dim=3)
    X = np.random.randn(2, 4) # Batch of 2, 4 features
    output = layer.forward(X)
    
    # Output should be (batch_size, output_dim)

    assert output.shape == (2, 3)

def test_relu_activation():
    #Verifies that ReLU zeroes out negative numbers and its gradient is a boolean mask
    
    layer = ReLU()
    
    X = np.array([[-1.0, 0.0, 2.0], [3.0, -5.0, 1.0]])
    
    # Forward check
    output = layer.forward(X)
    expected_output = np.array([[0.0, 0.0, 2.0], [3.0, 0.0, 1.0]])
    np.testing.assert_array_equal(output, expected_output)
    
    # Backward check (assume upstream gradient is all 1s)
    upstream_grad = np.ones_like(X)
    grad = layer.backward(upstream_grad)
    expected_grad = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    np.testing.assert_array_equal(grad, expected_grad)

def test_dense_numerical_gradient():
    """
    Finite-difference gradient check.
    Calculates derivative numerically and compares it to our analytical chain rule.
    """
    layer = Dense(input_dim=2, output_dim=2)
    X = np.random.randn(1, 2)
    
    # Get analytical gradients
    output = layer.forward(X)
    upstream_gradient = np.ones_like(output) # Simulate a basic loss gradient
    layer.backward(upstream_gradient)
    analytical_grad = layer.d_weights.copy()
    
    # Get numerical gradients using finite differences
    epsilon = 1e-5
    numerical_grad = np.zeros_like(layer.weights)
    
    for i in range(layer.weights.shape[0]):
        for j in range(layer.weights.shape[1]):
            # Add epsilon
            layer.weights[i, j] += epsilon
            out_plus = layer.forward(X)
            loss_plus = np.sum(out_plus) 
            
            # Subtract epsilon
            layer.weights[i, j] -= 2 * epsilon
            out_minus = layer.forward(X)
            loss_minus = np.sum(out_minus)
            
            # Restore original weight
            layer.weights[i, j] += epsilon
            
            # f(x + e) - f(x - e) / 2e
            numerical_grad[i, j] = (loss_plus - loss_minus) / (2 * epsilon)
            
    # Assert they are practically identical
    np.testing.assert_allclose(analytical_grad, numerical_grad, rtol=1e-4, atol=1e-4)