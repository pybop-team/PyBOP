import numpy as np

from pybop.costs._likelihoods import BaseLikelihood
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
        super(RootMeanSquaredError, self).__init__(problem)

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

        return float(e) if self.n_outputs == 1 else np.sum(e)

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
        super(SumSquaredError, self).__init__(problem)

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
            The sum of squared errors.
        """
        prediction = self.problem.evaluate(inputs)

        if not self.verify_prediction(prediction):
            return np.inf

        e = np.asarray(
            [
                np.sum(((prediction[signal] - self._target[signal]) ** 2))
                for signal in self.signal
            ]
        )

        return float(e) if self.n_outputs == 1 else np.sum(e)

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
    including Euclidean and Manhattan distances. It is defined as:

    .. math::
        L_p(x, y) = (\sum_i |x_i - y_i|^p)

    where p ≥ 1 is the order of the Minkowski metric.

    Special cases:
    - p = 1: Manhattan distance
    - p = 2: Euclidean distance
    - p → ∞: Chebyshev distance

    This class implements the Minkowski distance as a cost function for
    optimisation problems, allowing for flexible distance-based optimisation
    across various problem domains.

    Attributes:
        p (float): The order of the Minkowski metric.

    """

    def __init__(self, problem, p: float = 2.0):
        super(Minkowski, self).__init__(problem)
        if p < 0:
            raise ValueError(
                "The order of the Minkowski metric must be greater than 0."
            )
        self.p = p

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
                for signal in self.signal
            ]
        )

        return float(e) if self.n_outputs == 1 else np.sum(e)

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


class MAP(BaseLikelihood):
    """
    Maximum a posteriori cost function.

    Computes the maximum a posteriori cost function, which is the sum of the
    log likelihood and the log prior. The goal of maximising is achieved by
    setting minimising = False in the optimiser settings.

    Inherits all parameters and attributes from ``BaseLikelihood``.

    """

    def __init__(self, problem, likelihood, sigma=None):
        super(MAP, self).__init__(problem)
        self.sigma0 = sigma
        if self.sigma0 is None:
            self.sigma0 = []
            for param in self.problem.parameters:
                self.sigma0.append(param.prior.sigma)

        try:
            self.likelihood = likelihood(problem=self.problem, sigma=self.sigma0)
        except Exception as e:
            raise ValueError(
                f"An error occurred when constructing the Likelihood class: {e}"
            )

        if hasattr(self, "likelihood") and not isinstance(
            self.likelihood, BaseLikelihood
        ):
            raise ValueError(f"{self.likelihood} must be a subclass of BaseLikelihood")

    def _evaluate(self, inputs: Inputs, grad=None):
        """
        Calculate the maximum a posteriori cost for a given set of parameters.

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
            The maximum a posteriori cost.
        """
        log_likelihood = self.likelihood._evaluate(inputs)
        if not np.isfinite(log_likelihood):
            return -np.inf

        log_prior = sum(
            self.parameters[key].prior.logpdf(value) for key, value in inputs.items()
        )

        posterior = log_likelihood + log_prior
        return posterior

    def _evaluateS1(self, inputs: Inputs):
        """
        Compute the maximum a posteriori with respect to the parameters.
        The method passes the likelihood gradient to the optimiser without modification.

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
        log_likelihood, dl = self.likelihood._evaluateS1(inputs)
        if not np.isfinite(log_likelihood):
            return -np.inf, -dl

        log_prior = sum(
            self.parameters[key].prior.logpdf(inputs[key]) for key in inputs.keys()
        )
        posterior = log_likelihood + log_prior
        return posterior, dl
