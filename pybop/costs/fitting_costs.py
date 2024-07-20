import numpy as np

from pybop.costs.base_cost import BaseCost
from pybop.observers.observer import Observer
from pybop.parameters.parameter import Inputs


class RootMeanSquaredError(BaseCost):
    """
    Root mean square error cost function.

    Computes the root mean square error between model predictions and the target
    data, providing a measure of the differences between predicted values and
    observed values.

    Inherits all parameters and attributes from ``BaseCost``.

    """

    def __init__(self, problem):
        super().__init__(problem)

    def _evaluate(self, inputs: Inputs, grad=None):
        """
        Calculate the root mean square error for a given set of parameters.

        Parameters
        ----------
        inputs : Inputs
            The parameters for which to evaluate the cost.
        grad : array-like, optional
            An array to store the gradient of the cost function with respect
            to the parameters.

        Returns
        -------
        float
            The root mean square error.

        """
        prediction = self.problem.evaluate(inputs)

        if not self.verify_prediction(prediction):
            return np.inf

        e = np.asarray(
            [
                np.sqrt(np.mean((prediction[signal] - self._target[signal]) ** 2))
                for signal in self.signal
            ]
        )

        return e.item() if self.n_outputs == 1 else np.sum(e)

    def _evaluateS1(self, inputs: Inputs):
        """
        Compute the cost and its gradient with respect to the parameters.

        Parameters
        ----------
        inputs : Inputs
            The parameters for which to compute the cost and gradient.

        Returns
        -------
        tuple
            A tuple containing the cost and the gradient. The cost is a float,
            and the gradient is an array-like of the same length as `inputs`.

        Raises
        ------
        ValueError
            If an error occurs during the calculation of the cost or gradient.
        """
        y, dy = self.problem.evaluateS1(inputs)
        if not self.verify_prediction(y):
            return np.inf, self._de * np.ones(self.n_parameters)

        r = np.asarray([y[signal] - self._target[signal] for signal in self.signal])
        e = np.sqrt(np.mean(r**2, axis=1))
        de = np.mean((r * dy.T), axis=2) / (e + np.finfo(float).eps)

        if self.n_outputs == 1:
            return e.item(), de.flatten()
        else:
            return np.sum(e), np.sum(de, axis=1)


class SumSquaredError(BaseCost):
    """
    Sum of squared errors cost function.

    Computes the sum of the squares of the differences between model predictions
    and target data, which serves as a measure of the total error between the
    predicted and observed values.

    Inherits all parameters and attributes from ``BaseCost``.

    Additional Attributes
    ---------------------
    _de : float
        The gradient of the cost function to use if an error occurs during
        evaluation. Defaults to 1.0.

    """

    def __init__(self, problem):
        super().__init__(problem)

    def _evaluate(self, inputs: Inputs, grad=None):
        """
        Calculate the sum of squared errors for a given set of parameters.

        Parameters
        ----------
        inputs : Inputs
            The parameters for which to evaluate the cost.
        grad : array-like, optional
            An array to store the gradient of the cost function with respect
            to the parameters.

        Returns
        -------
        float
            The Sum of Squared Error.
        """
        prediction = self.problem.evaluate(inputs)

        if not self.verify_prediction(prediction):
            return np.inf

        e = np.asarray(
            [
                np.sum((prediction[signal] - self._target[signal]) ** 2)
                for signal in self.signal
            ]
        )

        return e.item() if self.n_outputs == 1 else np.sum(e)

    def _evaluateS1(self, inputs: Inputs):
        """
        Compute the cost and its gradient with respect to the parameters.

        Parameters
        ----------
        inputs : Inputs
            The parameters for which to compute the cost and gradient.

        Returns
        -------
        tuple
            A tuple containing the cost and the gradient. The cost is a float,
            and the gradient is an array-like of the same length as `inputs`.

        Raises
        ------
        ValueError
            If an error occurs during the calculation of the cost or gradient.
        """
        y, dy = self.problem.evaluateS1(inputs)
        if not self.verify_prediction(y):
            return np.inf, self._de * np.ones(self.n_parameters)

        r = np.asarray([y[signal] - self._target[signal] for signal in self.signal])
        e = np.sum(np.sum(r**2, axis=0), axis=0)
        de = 2 * np.sum(np.sum((r * dy.T), axis=2), axis=1)

        return e, de


class Minkowski(BaseCost):
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

    Attributes
    ----------
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

    def _evaluate(self, inputs: Inputs, grad=None):
        """
        Calculate the Minkowski cost for a given set of parameters.

        Parameters
        ----------
        inputs : Inputs
            The parameters for which to compute the cost and gradient.

        Returns
        -------
        float
            The Minkowski cost.
        """
        prediction = self.problem.evaluate(inputs)
        if not self.verify_prediction(prediction):
            return np.inf

        e = np.asarray(
            [
                np.sum(np.abs(prediction[signal] - self._target[signal]) ** self.p)
                ** (1 / self.p)
                for signal in self.signal
            ]
        )

        return e.item() if self.n_outputs == 1 else np.sum(e)

    def _evaluateS1(self, inputs):
        """
        Compute the cost and its gradient with respect to the parameters.

        Parameters
        ----------
        inputs : Inputs
            The parameters for which to compute the cost and gradient.

        Returns
        -------
        tuple
            A tuple containing the cost and the gradient. The cost is a float,
            and the gradient is an array-like of the same length as `inputs`.

        Raises
        ------
        ValueError
            If an error occurs during the calculation of the cost or gradient.
        """
        y, dy = self.problem.evaluateS1(inputs)
        if not self.verify_prediction(y):
            return np.inf, self._de * np.ones(self.n_parameters)

        r = np.asarray([y[signal] - self._target[signal] for signal in self.signal])
        e = np.asarray(
            [
                np.sum(np.abs(y[signal] - self._target[signal]) ** self.p)
                ** (1 / self.p)
                for signal in self.signal
            ]
        )
        de = np.sum(
            np.sum(r ** (self.p - 1) * dy.T, axis=2)
            / (e ** (self.p - 1) + np.finfo(float).eps),
            axis=1,
        )

        return np.sum(e), de


class SumofPower(BaseCost):
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

    Attributes:
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

    def _evaluate(self, inputs: Inputs, grad=None):
        """
        Calculate the Sum of Power cost for a given set of parameters.

        Parameters
        ----------
        inputs : Inputs
            The parameters for which to compute the cost and gradient.

        Returns
        -------
        float
            The Sum of Power cost.
        """
        prediction = self.problem.evaluate(inputs)
        if not self.verify_prediction(prediction):
            return np.inf

        e = np.asarray(
            [
                np.sum(np.abs(prediction[signal] - self._target[signal]) ** self.p)
                for signal in self.signal
            ]
        )

        return e.item() if self.n_outputs == 1 else np.sum(e)

    def _evaluateS1(self, inputs):
        """
        Compute the cost and its gradient with respect to the parameters.

        Parameters
        ----------
        inputs : Inputs
            The parameters for which to compute the cost and gradient.

        Returns
        -------
        tuple
            A tuple containing the cost and the gradient. The cost is a float,
            and the gradient is an array-like of the same length as `inputs`.

        Raises
        ------
        ValueError
            If an error occurs during the calculation of the cost or gradient.
        """
        y, dy = self.problem.evaluateS1(inputs)
        if not self.verify_prediction(y):
            return np.inf, self._de * np.ones(self.n_parameters)

        r = np.asarray([y[signal] - self._target[signal] for signal in self.signal])
        e = np.sum(np.sum(np.abs(r) ** self.p))
        de = self.p * np.sum(np.sum(r ** (self.p - 1) * dy.T, axis=2), axis=1)

        return e, de


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

    def _evaluate(self, inputs: Inputs, grad=None):
        """
        Calculate the observer cost for a given set of parameters.

        Parameters
        ----------
        inputs : Inputs
            The parameters for which to evaluate the cost.
        grad : array-like, optional
            An array to store the gradient of the cost function with respect
            to the parameters.

        Returns
        -------
        float
            The observer cost (negative of the log likelihood).
        """
        log_likelihood = self._observer.log_likelihood(
            self._target, self._observer.time_data(), inputs
        )
        return -log_likelihood

    def evaluateS1(self, inputs: Inputs):
        """
        Compute the cost and its gradient with respect to the parameters.

        Parameters
        ----------
        inputs : Inputs
            The parameters for which to compute the cost and gradient.

        Returns
        -------
        tuple
            A tuple containing the cost and the gradient. The cost is a float,
            and the gradient is an array-like of the same length as `inputs`.

        Raises
        ------
        ValueError
            If an error occurs during the calculation of the cost or gradient.
        """
        raise NotImplementedError
