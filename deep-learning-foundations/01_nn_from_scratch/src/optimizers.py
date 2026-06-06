import numpy as np

class SGDMomentum:
    """
    Stochastic Gradient Descent Optimizer with Momentum.
    Accelerates gradient updates in the correct direction and dampens oscillations.
    """

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        # Cache to store the current velocity for each layer's parameters
        self.velocities = {}


    def update(self, layers: list):
        """
        Iterates through the model's layers and updates trainable parameters.
        """
        for layer in layers:
            # We only update layers that actually have weights (like Dense)
            if layer.trainable:
                
                # Initialize velocities with zeros if this is the very first training step
                if layer not in self.velocities:
                    self.velocities[layer] = {
                        'weights': np.zeros_like(layer.weights),
                        'biases': np.zeros_like(layer.biases)
                    }


                # Update Weights

                # Velocity = (momentum * past_velocity) + (learning_rate * current_gradient)


                self.velocities[layer]['weights'] = (
                    self.momentum * self.velocities[layer]['weights'] 
                    + self.learning_rate * layer.d_weights
                )

                # Apply the update
                layer.weights -= self.velocities[layer]['weights']
                
                # Update Biases

                self.velocities[layer]['biases'] = (
                    self.momentum * self.velocities[layer]['biases'] 
                    + self.learning_rate * layer.d_biases
                )
                
                # Apply the update
                layer.biases -= self.velocities[layer]['biases']