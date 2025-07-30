from collections.abc import Callable

import numpy as np

from pybop.costs.base_cost import CallableCost


class CallableError(CallableCost):
    """
    Mean square error (MSE) cost function.

    Computes the mean square error between model predictions and the target
    data, providing a measure of the differences between predicted values and
    observed values.

    Parameters
    ----------
    weighting : np.ndarray, optional
        The type of weighting array to use when taking the sum or mean of the error
        measure. Options: "equal"(default), "domain", or a custom numpy array.
    """

    def __init__(self, callable_fun: Callable, weighting: str | np.ndarray = None):
        # should have two parameters: r and dy
        if not callable_fun or callable_fun.__code__.co_argcount not in (1, 2):
            raise ValueError(
                "Callable must accept one or two parameters: r and dy (optional)."
            )
        self._callable = callable_fun

    def __call__(
        self,
        r: np.ndarray,
        dy: np.ndarray | None = None,
    ) -> float | tuple[float, np.ndarray]:
        return self._callable(r, dy)


class MeanSquaredError(CallableCost):
    """
    Mean square error (MSE) cost function.

    Computes the mean square error between model predictions and the target
    data, providing a measure of the differences between predicted values and
    observed values.

    Parameters
    ----------
    weighting : np.ndarray, optional
        The type of weighting array to use when taking the sum or mean of the error
        measure. Options: "equal"(default), "domain", or a custom numpy array.
    """

    def __call__(
        self,
        r: np.ndarray,
        dy: np.ndarray | None = None,
    ) -> float | tuple[float, np.ndarray]:
        e = np.sum(np.abs(r) ** 2 * self.weighting)

        if dy is not None:
            de = 2 * np.sum((r * self.weighting) * dy)
            return e, de

        return e


class RootMeanSquaredError(CallableCost):
    """
    Root mean square error (RMSE) cost function.

    Computes the root mean square error between model predictions and the target
    data, providing a measure of the differences between predicted values and
    observed values.
    """

    def __call__(
        self,
        r: np.ndarray,
        dy: np.ndarray | None = None,
    ) -> float | tuple[float, np.ndarray]:
        e = np.sqrt(np.mean((np.abs(r) ** 2) * self.weighting))

        if dy is not None:
            de = np.mean((r * self.weighting) * dy) / (e + np.finfo(float).eps)
            return e, de

        return e


class MeanAbsoluteError(CallableCost):
    """
    Mean absolute error (MAE) cost function.

    Computes the mean absolute error (MAE) between model predictions
    and target data. The MAE is a measure of the average magnitude
    of errors in a set of predictions, without considering their direction.

    Parameters
    ----------
    weighting : np.ndarray, optional
        The type of weighting array to use when taking the sum or mean of the error
        measure. Options: "equal"(default), "domain", or a custom numpy array.
    """

    def __call__(
        self,
        r: np.ndarray,
        dy: np.ndarray | None = None,
    ) -> float | tuple[float, np.ndarray]:
        e = np.mean(np.abs(r) * self.weighting)

        if dy is not None:
            de = np.mean((np.sign(r) * self.weighting) * dy)
            return e, de

        return e


class SumSquaredError(CallableCost):
    """
    Sum of squared error (SSE) cost function.

    Computes the sum of the squares of the differences between model predictions
    and target data, which serves as a measure of the total error between the
    predicted and observed values.
    """

    def __call__(
        self,
        r: np.ndarray,
        dy: np.ndarray | None = None,
    ) -> float | tuple[float, np.ndarray]:
        e = np.sum(np.abs(r) ** 2 * self.weighting)

        if dy is not None:
            de = 2 * np.sum((r * self.weighting) * dy)
            return e, de

        return e


class Minkowski(CallableCost):
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

    def __init__(self, p: float = 2.0, weighting: str | np.ndarray = None):
        super().__init__(weighting=weighting)
        if p < 0:
            raise ValueError(
                "The order of the Minkowski distance must be greater than 0."
            )
        elif not np.isfinite(p):
            raise ValueError(
                "For p = infinity, an implementation of the Chebyshev distance is required."
            )
        self.p = float(p)

    def __call__(
        self,
        r: np.ndarray,
        dy: np.ndarray | None = None,
    ) -> float | tuple[float, np.ndarray]:
        e = np.sum((np.abs(r) ** self.p) * self.weighting) ** (1 / self.p)

        if dy is not None:
            de = np.sum(
                ((np.sign(r) * np.abs(r) ** (self.p - 1)) * self.weighting) * dy
            ) / (e ** (self.p - 1) + np.finfo(float).eps)
            return e, de

        return e


class SumOfPower(CallableCost):
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

    def __init__(self, p: float = 2.0, weighting: str | np.ndarray = None):
        super().__init__(weighting=weighting)
        if p < 0:
            raise ValueError("The order of 'p' must be greater than 0.")
        elif not np.isfinite(p):
            raise ValueError("p = np.inf is not yet supported.")
        self.p = float(p)

    def __call__(
        self,
        r: np.ndarray,
        dy: np.ndarray | None = None,
    ) -> float | tuple[float, np.ndarray]:
        e = np.sum((np.abs(r) ** self.p) * self.weighting)

        if dy is not None:
            de = self.p * np.sum(
                ((np.sign(r) * np.abs(r) ** (self.p - 1)) * self.weighting) * dy
            )
            return e, de

        return e
