from typing import Optional, Union

import numpy as np

from pybop import BaseCost


class FittingCost(BaseCost):
    """
    Overwrites and extends `BaseCost` class for fitting-type cost functions.

    This class is intended to be subclassed to create specific cost functions
    for evaluating model predictions against a set of data. The cost function
    quantifies the goodness-of-fit between the model predictions and the
    observed data, with a lower cost value indicating a better fit.

    Additional Parameters
    ---------------------
    weighting : Union[str, np.ndarray], optional
        The type of weighting to use when taking the sum or mean of the error
        measure.
    """

    def __init__(self, problem, weighting: Union[str, np.ndarray] = None):
        super().__init__(problem)
        self.weighting = None

        if weighting == "equal" or weighting is None:
            self.weighting = 1.0
        elif weighting == "domain":
            # Normalise the residuals by the domain spacing (for a uniform domain,
            # this is the same as a uniform weighting)
            domain_data = self.problem.domain_data
            domain_spacing = domain_data[1:] - domain_data[:-1]
            mean_spacing = np.mean(domain_spacing)
            self.weighting = np.concatenate(
                (
                    [(mean_spacing + domain_spacing[0]) / 2],
                    (domain_spacing[1:] + domain_spacing[:-1]) / 2,
                    [(domain_spacing[-1] + mean_spacing) / 2],
                )
            ) * ((len(domain_data) - 1) / (domain_data[-1] - domain_data[0]))
        else:
            self.weighting = np.asarray(weighting)

    def compute(
        self,
        y: dict,
        dy: Optional[dict] = None,
    ) -> Union[float, tuple[float, np.ndarray]]:
        """
        Computes the cost function for the given predictions.

        Parameters
        ----------
        y : dict[str, np.ndarray[np.float64]]
            The dictionary of predictions with keys designating the signals for fitting.
        dy : dict[str, dict[str, np.ndarray]], optional
            The corresponding sensitivities to each parameter for each signal.

        Returns
        -------
        np.float64 or tuple[np.float64, np.ndarray[np.float64]]
            If dy is not None, returns a tuple containing the cost (float) and the
            gradient with dimension (len(parameters)), otherwise returns only the cost.
        """
        # Early return if the prediction is not verified
        if not self.verify_prediction(y):
            return (np.inf, self.grad_fail) if dy is not None else np.inf

        # Compute the residual for all signals
        r = np.asarray([y[signal] - self._target[signal] for signal in self.signal])

        # Extract the sensitivities for all signals and parameters
        if dy is not None:
            dy = self.stack_sensitivities(dy)

        return self._error_measure(r=r, dy=dy)

    def _error_measure(
        self,
        r: np.ndarray,
        dy: Optional[np.ndarray] = None,
    ) -> Union[float, tuple[float, np.ndarray]]:
        """
        Computes the cost function for the given predictions.

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
