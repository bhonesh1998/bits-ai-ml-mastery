import numpy as np

class Sequential:
    """
    A container/manager that stacks layers sequentially. 
    Handles the global forward and backward passes across the entire network.
    """

    def __init__(self):
        self.layers = [] # list of layers
        self.loss_function = None

    def add(self, layer):
        """Appends a layer to the network architecture."""
        self.layers.append(layer)

    def compile(self, loss_function):
        """Sets up the loss function for training."""
        self.loss_function = loss_function

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Passes the input data through every layer sequentially.
        The output of layer L becomes the input to layer L+1.
        It is like pipelining.
        """
        
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, predictions: np.ndarray, targets: np.ndarray):
        """
        Executes backpropagation. First computes the loss gradient, 
        then passes it backward through the layers in reverse order.
        """
        # 1. Get the gradient from the loss function
        gradient = self.loss_function.backward()
        
        # 2. Propagate the gradient backwards through the network
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)