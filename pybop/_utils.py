import numpy as np


def is_numeric(x):
    """
    Check if a variable is numeric.
    """
    return isinstance(x, (int, float, np.number))
