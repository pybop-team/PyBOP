from typing import Optional, Union

import numpy as np


class BaseCost:
    """
    Base class for defining cost functions.

    Parameters
    ----------
    weighting : np.ndarray, optional
        The type of weighting array to use when taking the sum or mean of the error
        measure.
    """

    def __init__(self, weighting: np.ndarray = None):
        if weighting is None or weighting == "equal":
            self.weighting = 1.0
        elif weighting == "domain":
            self._set_domain_weighting()
        else:
            self.weighting = np.asarray(weighting)

    def _set_domain_weighting(self):
        """Calculate domain-based weighting."""
        domain_data = self.problem.domain_data
        domain_spacing = domain_data[1:] - domain_data[:-1]
        mean_spacing = np.mean(domain_spacing)

        # Create weights array in one operation
        self.weighting = np.concatenate(
            (
                [(mean_spacing + domain_spacing[0]) / 2],
                (domain_spacing[1:] + domain_spacing[:-1]) / 2,
                [(domain_spacing[-1] + mean_spacing) / 2],
            )
        ) * ((len(domain_data) - 1) / (domain_data[-1] - domain_data[0]))

    @staticmethod
    def __call__(
        r: np.ndarray,
        dy: Optional[np.ndarray] = None,
    ) -> Union[float, tuple[float, np.ndarray]]:
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
