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
            return np.sqrt(np.mean((prediction - target) ** 2))

        except Exception as e:
            print(f"Error in RMSE calculation: {e}")
            return None


class MLE:
    """
    Defines the cost function for maximum likelihood estimation.
    """

    def __init__(self):
        self.name = "MLE"

    def compute(self, prediction, target):
        # Compute the cost
        try:
            return 0  # update with MLE residual

        except Exception as e:
            print(f"Error in RMSE calculation: {e}")
            return None


class PEM:
    """
    Defines the cost function for prediction error minimisation.
    """

    def __init__(self):
        self.name = "PEM"

    def compute(self, prediction, target):
        # Compute the cost
        try:
            return 0  # update with MLE residual

        except Exception as e:
            print(f"Error in RMSE calculation: {e}")
            return None


class MAP:
    """
    Defines the cost function for maximum a posteriori estimation.
    """

    def __init__(self):
        self.name = "MAP"

    def compute(self, prediction, target):
        # Compute the cost
        try:
            return 0  # update with MLE residual

        except Exception as e:
            print(f"Error in RMSE calculation: {e}")
            return None

    def sample(self, n_chains):
        """
        Sample from the posterior distribution.
        """
        pass
