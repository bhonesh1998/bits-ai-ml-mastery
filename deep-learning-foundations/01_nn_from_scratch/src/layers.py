import numpy as np

class Layer:
    
    """
    Abstract Base Class for all neural network layers.
    Every layer in our network will inherit from this class, 
    ensuring they all have a standardized forward and backward pass for computation.
    """

    def __init__(self):
        # By default, layers do not have trainable parameters
        self.trainable = False

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Processes input data and returns the layer's output
        ndarray means N dimensional array
        """
        raise NotImplementedError

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        """Receives the gradient from the next layer and computes local gradients."""
        raise NotImplementedError


class Dense(Layer):
    """
    Fully Connected (Linear) Layer.
    Mathematical operation
    Y = XW + b
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        # A Dense layer has Weights and Biases that need to be updated during training
        self.trainable = True
        
        # --- WEIGHT INITIALIZATION ---
        # Use Kaiming Initialization to prevent the vanishing/exploding 
        # gradient problem. It scales random normal weights by sqrt(2 / input_dim).

        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        
        # Biases are typically initialized to zero
        self.biases = np.zeros((1, output_dim))
        
        # --- MEMORY CACHE ---
        # We must remember the input data (X) during the forward pass because 
        # we need it to calculate the gradient with respect to the weights later.
        self.input_data = None
        
        # --- GRADIENT STORAGE ---
        # Optimizers (like SGD or Adam) will read these arrays to update the weights
        self.d_weights = None
        self.d_biases = None


    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Forward pass execution.
        Input (X): shape (batch_size, input_dim)
        Weights (W): shape (input_dim, output_dim)
        Biases (b): shape (1, output_dim)
        
        Returns Y = XW + b of shape (batch_size, output_dim)
        """
        self.input_data = input_data  # Cache X for the backward pass
        
        # np.dot(X, W) performs the batch matrix multiplication
        # + self.biases uses NumPy broadcasting to add the bias vector to every row

        return np.dot(input_data, self.weights) + self.biases


    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        """
        Backward pass using the chain rule.
        output_gradient (dL/dY): The gradient of the loss with respect to this layer's output.
        Shape: (batch_size, output_dim)
        """
        
        # Gradient with respect to Weights (dL/dW)
        # Math: dL/dW = X^T * dL/dY
        # Why transpose X? To align the dimensions for the matrix dot product.
        # Shape check: (input_dim, batch_size) dot (batch_size, output_dim) = (input_dim, output_dim)
        self.d_weights = np.dot(self.input_data.T, output_gradient)
        
        # Gradient with respect to Biases (dL/db)
        # Math: dL/db = Sum of dL/dY across the batch dimension
        # Since the same bias was added to every sample in the batch, its gradient 
        # is the sum of gradients from all those samples.
        self.d_biases = np.sum(output_gradient, axis=0, keepdims=True)
        
        # Gradient with respect to Inputs (dL/dX)
        # Math: dL/dX = dL/dY * W^T
        # We return this so the previous layer in the network can use it as ITS output_gradient.
        # Shape check: (batch_size, output_dim) dot (output_dim, input_dim) = (batch_size, input_dim)
        return np.dot(output_gradient, self.weights.T)

# For non activation features , introducing RELU and softmax 

class ReLU(Layer):
    """
    Rectified Linear Unit (ReLU) Activation Layer.
    This introduces non-linearity, allowing the network to learn complex patterns.
    This is important to know why non-linearity is introduces in network.
    Mathematical operation: f(x) = max(0, x)
    """
    def __init__(self):
        super().__init__()
        self.input_data = None

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Any negative number becomes 0. Any positive number stays the same.
        """
        self.input_data = input_data
        return np.maximum(0, input_data)

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        """
        The derivative of ReLU is 1 if x > 0, and 0 if x <= 0.
        By the chain rule, we just multiply the upstream gradient by this derivative.
        """
        relu_derivative = self.input_data > 0
        return output_gradient * relu_derivative


# softmax is used for probability distribution

class Softmax(Layer):
    """
    Softmax Activation Layer.
    Used at the very end of a classification network to convert raw scores (logits)
    into a beautiful probability distribution that sums to 1.0.
    """
    def __init__(self):
        super().__init__()
        self.output_data = None

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Computes the exponential of each element, divided by the sum of exponentials.
        Includes a critical numerical stability trick to prevent np.exp() from overflowing.
        """

        # Subtract the max value in each row to prevent exponent overflow
        shifted_input = input_data - np.max(input_data, axis=1, keepdims=True)
        
        # Calculate exponentials and probabilities
        exponentials = np.exp(shifted_input)
        self.output_data = exponentials / np.sum(exponentials, axis=1, keepdims=True)
        return self.output_data

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        """
        The Jacobian matrix of Softmax is complex because each output depends on ALL inputs.
        Math: dL/dX = S * (dL/dY - sum(dL/dY * S))
        """
        # This is a highly vectorized, computationally efficient way to calculate 
        # the backward pass for Softmax across a batch of data.
        
        dot_product_sum = np.sum(output_gradient * self.output_data, axis=1, keepdims=True)
        return self.output_data * (output_gradient - dot_product_sum)