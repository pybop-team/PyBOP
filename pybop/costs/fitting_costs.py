from typing import Optional, Union

import numpy as np

from pybop.costs.base_cost import BaseCost
from pybop.observers.observer import Observer


class FittingCost(BaseCost):
    """
    Overwrites and extends `BaseCost` class for fitting-type cost functions.

    This class is intended to be subclassed to create specific cost functions
    for evaluating model predictions against a set of data. The cost function
    quantifies the goodness-of-fit between the model predictions and the
    observed data, with a lower cost value indicating a better fit.
    """

    def __init__(self, problem):
        super().__init__(problem)
        self.numpy_axis = (0, 2) if self.n_outputs > 1 else (1, 2)

    def compute(
        self,
        y: dict,
        dy: Optional[np.ndarray] = None,
    ) -> Union[float, tuple[float, np.ndarray]]:
        """
        Computes the cost function for the given predictions.

        Parameters
        ----------
        y : dict
            The dictionary of predictions with keys designating the signals for fitting.
        dy : np.ndarray, optional
            The corresponding gradient with respect to the parameters for each signal.

        Returns
        -------
        tuple or float
            If dy is not None, returns a tuple containing the cost (float) and the
            gradient (np.ndarray), otherwise returns only the computed cost (float).
        """
        # Early return if the prediction is not verified
        if not self.verify_prediction(y):
            return (np.inf, self.grad_fail) if dy is not None else np.inf

        # Compute the residual
        r = np.asarray([y[signal] - self._target[signal] for signal in self.signal])

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
            The residual difference between the model prediction and the target.
        dy : np.ndarray, optional
            The corresponding gradient with respect to the parameters for each signal.

        Returns
        -------
        tuple or float
            If dy is not None, returns a tuple containing the cost (float) and the
            gradient (np.ndarray), otherwise returns only the computed cost (float).
        """
        raise NotImplementedError


class MeanSquaredError(FittingCost):
    """
    Mean square error (MSE) cost function.

    Computes the mean square error between model predictions and the target
    data, providing a measure of the differences between predicted values and
    observed values.
    """

    def _error_measure(
        self,
        r: np.ndarray,
        dy: Optional[np.ndarray] = None,
    ) -> Union[float, tuple[float, np.ndarray]]:
        e = np.mean(np.abs(r) ** 2)

        if dy is not None:
            de = 2 * np.mean((r * dy.T), axis=self.numpy_axis)
            return e, de

        return e


class RootMeanSquaredError(FittingCost):
    """
    Root mean square error (RMSE) cost function.

    Computes the root mean square error between model predictions and the target
    data, providing a measure of the differences between predicted values and
    observed values.
    """

    def _error_measure(
        self,
        r: np.ndarray,
        dy: Optional[np.ndarray] = None,
    ) -> Union[float, tuple[float, np.ndarray]]:
        e = np.sqrt(np.mean(np.abs(r) ** 2))

        if dy is not None:
            de = np.mean((r * dy.T), axis=self.numpy_axis) / (e + np.finfo(float).eps)
            return e, de

        return e


class MeanAbsoluteError(FittingCost):
    """
    Mean absolute error (MAE) cost function.

    Computes the mean absolute error (MAE) between model predictions
    and target data. The MAE is a measure of the average magnitude
    of errors in a set of predictions, without considering their direction.
    """

    def _error_measure(
        self,
        r: np.ndarray,
        dy: Optional[np.ndarray] = None,
    ) -> Union[float, tuple[float, np.ndarray]]:
        e = np.mean(np.abs(r))

        if dy is not None:
            sign_r = np.sign(r)
            de = np.mean(sign_r * dy.T, axis=self.numpy_axis)
            return e, de

        return e


class SumSquaredError(FittingCost):
    """
    Sum of squared error (SSE) cost function.

    Computes the sum of the squares of the differences between model predictions
    and target data, which serves as a measure of the total error between the
    predicted and observed values.
    """

    def _error_measure(
        self,
        r: np.ndarray,
        dy: Optional[np.ndarray] = None,
    ) -> Union[float, tuple[float, np.ndarray]]:
        e = np.sum(np.abs(r) ** 2)

        if dy is not None:
            de = 2 * np.sum((r * dy.T), axis=self.numpy_axis)
            return e, de

        return e


class Minkowski(FittingCost):
    """
    The Minkowski distance is a generalisation of several distance metrics,
    including the Euclidean and Manhattan distances. It is defined as:

    .. math::
        L_p(x, y) = ( \\sum_i |x_i - y_i|^p )^(1/p)

    where p > 0 is the order of the Minkowski distance. For p ≥ 1, the
    Minkowski distance is a metric. For 0 < p < 1, it is not a metric, as it
    does not satisfy the triangle inequality, although a metric can be
    obtained by removing the (1/p) exponent.

    Special cases:

    * p = 1: Manhattan distance
    * p = 2: Euclidean distance
    * p → ∞: Chebyshev distance (not implemented as yet)

    This class implements the Minkowski distance as a cost function for
    optimisation problems, allowing for flexible distance-based optimisation
    across various problem domains.

    Additional Attributes
    ---------------------
    p : float, optional
        The order of the Minkowski distance.
    """

    def __init__(self, problem, p: float = 2.0):
        super().__init__(problem)
        if p < 0:
            raise ValueError(
                "The order of the Minkowski distance must be greater than 0."
            )
        elif not np.isfinite(p):
            raise ValueError(
                "For p = infinity, an implementation of the Chebyshev distance is required."
            )
        self.p = float(p)

    def _error_measure(
        self,
        r: np.ndarray,
        dy: Optional[np.ndarray] = None,
    ) -> Union[float, tuple[float, np.ndarray]]:
        e = np.sum(np.abs(r) ** self.p) ** (1 / self.p)

        if dy is not None:
            de = np.sum(
                np.sign(r) * np.abs(r) ** (self.p - 1) * dy.T, axis=self.numpy_axis
            ) / (e ** (self.p - 1) + np.finfo(float).eps)
            return e, de

        return e


class SumofPower(FittingCost):
    """
    The Sum of Power [1] is a generalised cost function based on the p-th power
    of absolute differences between two vectors. It is defined as:

    .. math::
        C_p(x, y) = \\sum_i |x_i - y_i|^p

    where p ≥ 0 is the power order.

    This class implements the Sum of Power as a cost function for
    optimisation problems, allowing for flexible power-based optimisation
    across various problem domains.

    Special cases:

    * p = 1: Sum of Absolute Differences
    * p = 2: Sum of Squared Differences
    * p → ∞: Maximum Absolute Difference

    Note that this is not normalised, unlike distance metrics. To get a
    distance metric, you would need to take the p-th root of the result.

    [1]: https://mathworld.wolfram.com/PowerSum.html

    Additional Attributes
    ---------------------
    p : float, optional
        The power order for Sum of Power.
    """

    def __init__(self, problem, p: float = 2.0):
        super().__init__(problem)
        if p < 0:
            raise ValueError("The order of 'p' must be greater than 0.")
        elif not np.isfinite(p):
            raise ValueError("p = np.inf is not yet supported.")
        self.p = float(p)

    def _error_measure(
        self,
        r: np.ndarray,
        dy: Optional[np.ndarray] = None,
    ) -> Union[float, tuple[float, np.ndarray]]:
        e = np.sum(np.abs(r) ** self.p)

        if dy is not None:
            de = self.p * np.sum(
                np.sign(r) * np.abs(r) ** (self.p - 1) * dy.T, axis=self.numpy_axis
            )
            return e, de

        return e


class ObserverCost(BaseCost):
    """
    Observer cost function.

    Computes the cost function for an observer model, which is log likelihood
    of the data points given the model parameters.

    Inherits all parameters and attributes from ``BaseCost``.
    """

    def __init__(self, observer: Observer):
        super().__init__(problem=observer)
        self._observer = observer
        self._has_separable_problem = False

    def compute(
        self,
        y: dict,
        dy: Optional[np.ndarray] = None,
    ) -> float:
        """
        Computes the cost function for the given predictions.

        Parameters
        ----------
        y : dict
            The dictionary of predictions with keys designating the signals for fitting.
        dy : np.ndarray, optional
            The corresponding gradient with respect to the parameters for each signal.

        Returns
        -------
        float
            The observer cost (negative of the log likelihood).
        """
        inputs = self._parameters.as_dict()
        log_likelihood = self._observer.log_likelihood(
            self._target, self._observer.domain_data, inputs
        )
        return -log_likelihood
