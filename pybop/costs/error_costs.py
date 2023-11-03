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

        # Compute the cost
        try:
            res = np.sqrt(np.mean((prediction - target) ** 2))
            print("Cost:", res)
            return res

        except Exception as e:
            raise ValueError(f"Error in RMSE calculation: {e}")


class MLE:
    """
    Defines the cost function for maximum likelihood estimation.
    """

    def __init__(self):
        self.name = "MLE"

    def compute(self, prediction, target):
        # Compute the cost
        return 0  # update with MLE residual


class PEM:
    """
    Defines the cost function for prediction error minimisation.
    """

    def __init__(self):
        self.name = "PEM"

    def compute(self, prediction, target):
        # Compute the cost
        return 0  # update with MLE residual


class MAP:
    """
    Defines the cost function for maximum a posteriori estimation.
    """

    def __init__(self):
        self.name = "MAP"

    def compute(self, prediction, target):
        # Compute the cost
        return 0  # update with MLE residual

    def sample(self, n_chains):
        """
        Sample from the posterior distribution.
        """
        pass
