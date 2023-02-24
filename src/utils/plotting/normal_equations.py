import numpy as np


__all__ = [
    'linear',
    'trigonometric',
]

def linear(x: np.array, y: np.array) -> np.array:
    """calculates `theta = (A.T * A)^-1 * A.T * b`, with `A_i = [1, x_i]`

    Args:
        x (np.array): x-values, supposed to be an one-dimensional array.
        y (np.array): corresponding y-values. 

    Returns:
        np.array: Parameter for line of best fit (given by y = theta[0] + theta[1] * x).
    """
    assert len(x.shape) == 1 or x.shape[1] == 1
    assert len(y.shape) == 1 or y.shape[1] == 1

    # add a column of 1s to X for the intercept term
    X = np.column_stack((np.ones(len(x)), x))

    # calculate theta using the normal equation
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    return theta


def trigonometric(x: np.array, y: np.array) -> np.array:
    """Calculates `theta = (A.T * A)^-1 * A.T * b` for non-linear/trigonometric data.
    
    with `A_i = [1, sin(x_i), cos(x_i)]`. Then the fitted curve becomes
        `y = theta[0] + theta[1] * sin(x) + theta[2] * cos(x)`.
    When the AE should reconstruct a sine-curve, the parameters should be `theta ~= [0, 1, 0]`.

    Args:
        x (np.array): x-values, supposed to be an one-dimensional array.
        y (np.array): corresponding y-values. 

    Returns:
        np.array: Parameter for curve of best fit.
    """
    # Transform the variables
    X = np.column_stack((np.ones(len(x)), np.sin(x), np.cos(x)))

    # Solve the normal equation to find theta
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    return theta
