"""
Runs neural network on: 
1) Friedman #1
2) Friedman #2
3) Friedman #3

Angie McGraw
Last Updated: March 28th, 2026
"""

import numpy as np
from utils.data_loader import load_dataset
from utils.plotting import plot_double_descent
from models.neural_network import NeuralNetwork

# Load dataset 
X_train, X_test, y_train, y_test = load_dataset("friedman1")

# Vary complexity (hidden layer size)
complexities = [1, 2, 5, 10, 20, 50, 100, 200, 500]
train_errors = []
test_errors = []

for c in complexities:
    nn_model = NeuralNetwork(input_dim=X_train.shape[1], complexity=c, epochs=200)
    nn_model.fit(X_train, y_train)

    y_train_pred = nn_model.predict(X_train)
    y_test_pred = nn_model.predict(X_test)

    train_mse = np.mean((y_train - y_train_pred)**2)
    test_mse = np.mean((y_test - y_test_pred)**2)

    train_errors.append(train_mse)
    test_errors.append(test_mse)

# Plot individual NN double descent
plot_double_descent(complexities, train_errors, test_errors, model_name="Neural Network")