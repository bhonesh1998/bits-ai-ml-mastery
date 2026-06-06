import numpy as np

class CategoricalCrossEntropy:
    """
    Measures the performance of a classification model where the output is a probability value between 0 and 1.
    """

    def __init__(self):
        self.predictions = None
        self.targets = None

    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Computes the cross-entropy loss.
        Math: L = - (1/N) * sum(targets * log(predictions))
        """
        # Clip predictions to prevent taking log(0) which results in negative infinity
        predictions_clipped = np.clip(predictions, 1e-7, 1 - 1e-7)
        self.predictions = predictions_clipped
        self.targets = targets
        
        samples = len(predictions)

        # Calculate negative log likelihoods

        correct_confidences = np.sum(targets * np.log(predictions_clipped), axis=1)
        loss = -np.mean(correct_confidences)
        return loss


    def backward(self) -> np.ndarray:
        """
        Computes the gradient of the loss with respect to the predictions.
        Math: dL/dY_pred = - (targets / predictions) / N
        """
        samples = len(self.predictions)
        
        # Gradient formula scaled by the batch size

        d_inputs = -self.targets / self.predictions
        d_inputs = d_inputs / samples
        return d_inputs