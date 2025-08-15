"""Simple gradient descent implementation for linear regression.

This module provides functions to compute the cost of a linear regression
model and to optimize the model parameters using gradient descent.

The implementation is intentionally lightweight and has no external
dependencies beyond NumPy. It can serve as a reference or a starting point
for experiments with supervised learning algorithms.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


def compute_cost(x: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    """Compute mean squared error cost for linear regression.

    Parameters
    ----------
    x:
        Matrix of shape (m, n) representing the input features. It is
        expected that the first column of ``x`` is all ones if an intercept
        term is desired.
    y:
        Vector of shape (m,) containing the target values.
    theta:
        Vector of shape (n,) containing the model parameters.

    Returns
    -------
    float
        The value of the cost function for the given parameters.
    """

    m = y.size
    predictions = x @ theta
    errors = predictions - y
    return float((errors @ errors) / (2 * m))


def gradient_descent(
    x: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
    learning_rate: float = 0.01,
    iterations: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run batch gradient descent to fit ``theta`` for linear regression.

    Parameters
    ----------
    x:
        Matrix of shape (m, n) representing the input features.
    y:
        Vector of shape (m,) containing the target values.
    theta:
        Initial parameter vector of shape (n,).
    learning_rate:
        Step size used when updating ``theta``.
    iterations:
        Number of iterations to run the optimization.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the optimized ``theta`` and an array with the
        value of the cost function at each iteration.
    """

    theta = theta.astype(float)
    m = y.size
    cost_history = np.zeros(iterations)

    for i in range(iterations):
        predictions = x @ theta
        errors = predictions - y
        gradient = (x.T @ errors) / m
        theta -= learning_rate * gradient
        cost_history[i] = compute_cost(x, y, theta)

    return theta, cost_history


if __name__ == "__main__":
    # Example usage with a simple linear relationship
    # y = 2x + 1
    x = np.array([[1, 0], [1, 1], [1, 2], [1, 3]], dtype=float)
    y = np.array([1, 3, 5, 7], dtype=float)
    initial_theta = np.zeros(x.shape[1])
    theta, history = gradient_descent(x, y, initial_theta, learning_rate=0.1, iterations=1000)
    print("Learned parameters:", theta)
    print("Final cost:", history[-1])
