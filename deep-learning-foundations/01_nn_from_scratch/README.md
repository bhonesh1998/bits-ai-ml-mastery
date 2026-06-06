# 01. Neural Networks From Scratch

## Objective
The goal of this module is to build a fully functional Deep Neural Network using pure **NumPy**. By avoiding auto-differentiation frameworks like PyTorch or TensorFlow, this code demonstrates a rigorous mathematical understanding of forward propagation, matrix calculus, and the backpropagation algorithm.

## The Mathematical Foundation
At the core of this network is the Dense (Fully Connected) layer, defined by the linear mapping equation:

$$Y = XW + b$$

During backpropagation, we calculate the gradients analytically using the chain rule:

* **Weights Gradient:** $\frac{\partial L}{\partial W} = X^T \cdot \frac{\partial L}{\partial Y}$
* **Bias Gradient:** $\frac{\partial L}{\partial b} = \sum \frac{\partial L}{\partial Y}$
* **Input Gradient:** $\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \cdot W^T$

## Directory Architecture
```text
01_nn_from_scratch/
├── configs/              # YAML hyperparameter settings
├── src/
│   ├── layers.py         # Dense, Activation, and Dropout architectures
│   ├── optimizers.py     # SGD with Momentum and Adam
│   └── model.py          # Sequential network logic
├── tests/
│   └── test_gradients.py # PyTest numerical finite-difference checks
└── train.py              # Main training loop execution