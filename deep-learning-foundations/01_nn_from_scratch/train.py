import numpy as np
import matplotlib.pyplot as plt

from src.layers import Dense, ReLU, Softmax
from src.losses import CategoricalCrossEntropy
from src.model import Sequential
from src.optimizers import SGDMomentum


# DATA GENERATOR
def create_spiral_data(points: int, classes: int):
    """
    Generates a non-linear spiral dataset.
    Returns X (features) and y (one-hot encoded labels).
    """
    X = np.zeros((points * classes, 2))
    y = np.zeros((points * classes, classes))
    
    for class_number in range(classes):
        ix = range(points * class_number, points * (class_number + 1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number * 4, (class_number + 1) * 4, points) + np.random.randn(points) * 0.2
        X[ix] = np.c_[r * np.sin(t*2.5), r * np.cos(t*2.5)]
        y[ix, class_number] = 1 # One-hot encoding
        
    return X, y


# VISUALIZATION FUNCTION
def plot_decision_boundary(model, X, y):
    """
    Plots the dataset and the network's learned decision boundary.
    """
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    predictions = model.forward(grid_points)
    Z = np.argmax(predictions, axis=1)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Spectral)
    
    true_classes = np.argmax(y, axis=1)
    plt.scatter(X[:, 0], X[:, 1], c=true_classes, s=40, cmap=plt.cm.Spectral, edgecolors='k')
    
    plt.title("Neural Network Decision Boundary")
    plt.show()


# Generate 100 points per class, 3 classes
X_train, y_train = create_spiral_data(100, 3)


# BUILD THE NETWORK
model = Sequential()


# Input features = 2 (x, y coordinates) -> Hidden Layer = 64
model.add(Dense(2, 64))
model.add(ReLU())


# Hidden Layer = 64 -> Output Layer = 3 (number of classes)
model.add(Dense(64, 3))
model.add(Softmax())


# COMPILE & CONFIGURE
loss_function = CategoricalCrossEntropy()
model.compile(loss_function)


optimizer = SGDMomentum(learning_rate=1.0, momentum=0.9)


# THE TRAINING LOOP
epochs = 10001

print(" Starting Training Phase ... \n")


for epoch in range(epochs):
    # Forward pass
    predictions = model.forward(X_train)
    
    # Calculate Loss
    loss = model.loss_function.forward(predictions, y_train)
    
    # Calculate Accuracy
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_train, axis=1)
    accuracy = np.mean(predicted_classes == true_classes)
    
    # Backward pass (Calculate gradients)
    model.backward(predictions, y_train)
    
    # Optimize (Update weights and biases)
    optimizer.update(model.layers)
    
    # Print metrics every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch: {epoch:05d} | Loss: {loss:.4f} | Accuracy: {accuracy:.4f}")

print("\n Training Complete! ")
plot_decision_boundary(model, X_train, y_train)