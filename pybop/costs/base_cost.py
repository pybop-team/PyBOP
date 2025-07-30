import numpy as np


class CallableCost:
    """
    Base class for defining cost functions.

    Parameters
    ----------
    weighting : np.ndarray, optional
        The type of weighting array to use when taking the sum or mean of the error
        measure. Options: "equal"(default), "domain", or a custom numpy array.
    """

    def __init__(self, weighting: str | np.ndarray = None):
        self.weighting = weighting

    def __call__(
        self,
        r: np.ndarray,
        dy: np.ndarray | None = None,
    ) -> float | tuple[float, np.ndarray]:
        """
        Computes the cost function for the given residuals.

        Parameters
        ----------
        r : np.ndarray
            The residual difference between the model prediction and the target. The
            dimensions of r are (len(signal), len(domain_data)).
        dy : np.ndarray, optional
            The corresponding gradient with respect to the parameters for each signal.
            The dimensions of dy are (len(parameters), len(signal), len(domain_data)).

        Returns
        -------
        np.float64 or tuple[np.float64, np.ndarray[np.float64]]
            If dy is not None, returns a tuple containing the cost (float) and the
            gradient with dimension (len(parameters)), otherwise returns only the cost.
        """
        raise NotImplementedError
