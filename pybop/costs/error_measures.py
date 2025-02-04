from typing import Optional, Union

import numpy as np

from pybop import FittingCost


class MeanFittingCost(FittingCost):
    """
    A subclass for fitting costs which involve taking the mean with respect to
    the domain data.
    """

    def __init__(self, problem):
        super().__init__(problem)
        self.domain_data = None
        if self.problem is not None:
            x = self.problem.domain_data
            if np.all(x[:-1] <= x[1:]):
                self.domain_data = x / (x[-1] - x[0]) * (len(x) - 1)


class MeanSquaredError(MeanFittingCost):
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
        L = r.shape[-1] - 1
        e = np.mean(np.trapz(np.abs(r) ** 2, self.domain_data) / L)

        if dy is not None:
            de = 2 * np.mean(
                np.trapz((r * dy.T), self.domain_data, axis=2) / L,
                axis=self.output_axis,
            )
            return e, de

        return e


class RootMeanSquaredError(MeanFittingCost):
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
        L = r.shape[-1] - 1
        e = np.sqrt(np.mean(np.trapz(np.abs(r) ** 2, self.domain_data) / L))

        if dy is not None:
            de = np.mean(
                np.trapz((r * dy.T), self.domain_data, axis=2) / L,
                axis=self.output_axis,
            ) / (e + np.finfo(float).eps)
            return e, de

        return e


class MeanAbsoluteError(MeanFittingCost):
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
        L = r.shape[-1] - 1
        e = np.mean(np.trapz(np.abs(r), self.domain_data) / L)

        if dy is not None:
            sign_r = np.sign(r)
            de = np.mean(
                np.trapz(sign_r * dy.T, self.domain_data, axis=2) / L,
                axis=self.output_axis,
            )
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
            de = 2 * np.sum((r * dy.T), axis=(self.output_axis, 2))
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
                np.sign(r) * np.abs(r) ** (self.p - 1) * dy.T,
                axis=(self.output_axis, 2),
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
                np.sign(r) * np.abs(r) ** (self.p - 1) * dy.T,
                axis=(self.output_axis, 2),
            )
            return e, de

        return e
