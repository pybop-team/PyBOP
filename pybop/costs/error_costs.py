import pybop
import numpy as np


class RMSE:
    """
    Defines the root mean square error cost function.
    """

    def __init__(self):
        self.name = "RMSE"

    def compute(self, prediction, target):
        # Check compatibility
        if len(prediction) != len(target):
            print(
                "Length of vectors:",
                len(prediction),
                len(target),
            )
            raise ValueError(
                "Measurement and simulated data length mismatch, potentially due to reaching a voltage cut-off"
            )

        print("Last Values:", prediction[-1], target[-1])

        # Compute the cost
        try:
            cost = np.sqrt(np.mean((prediction - target) ** 2))
        except:
            print("Error in RMSE calculation")

        return cost


class MLE:
    """
    Defines the cost function for maximum likelihood estimation.
    """

    def __init__(self):
        self.name = "MLE"

    def compute(self, prediction, target):
        # Compute the cost
        try:
            cost = 0  # update with MLE residual
        except:
            print("Error in MLE calculation")

        return cost


class PEM:
    """
    Defines the cost function for prediction error minimisation.
    """

    def __init__(self):
        self.name = "PEM"

    def compute(self, prediction, target):
        # Compute the cost
        try:
            cost = 0  # update with MLE residual
        except:
            print("Error in PEM calculation")

        return cost


class MAP:
    """
    Defines the cost function for maximum a posteriori estimation.
    """

    def __init__(self):
        self.name = "MAP"

    def compute(self, prediction, target):
        # Compute the cost
        try:
            cost = 0  # update with MLE residual
        except:
            print("Error in MAP calculation")

        return cost

    def sample(self, n_chains):
        """
        Sample from the posterior distribution.
        """
        pass
