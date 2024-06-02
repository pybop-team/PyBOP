from typing import List, Tuple, Union

import numpy as np

from pybop.costs.base_cost import BaseCost
from pybop.problems.base_problem import BaseProblem


class BaseLikelihood(BaseCost):
    """
    Base class for likelihoods
    """

    def __init__(self, problem: BaseProblem, sigma: Union[None, np.ndarray] = None):
        super(BaseLikelihood, self).__init__(problem, sigma)
        self.n_time_data = problem.n_time_data

    def set_sigma(self, sigma: Union[np.ndarray, List[float]]):
        """
        Setter for sigma parameter
        """
        sigma = np.asarray(sigma, dtype=float)
        if not np.all(sigma > 0):
            raise ValueError("Sigma must be positive")
        self.sigma0 = sigma

    def get_sigma(self) -> np.ndarray:
        """
        Getter for sigma parameter
        """
        return self.sigma0

    def get_n_parameters(self) -> int:
        """
        Returns the number of parameters
        """
        return self._n_parameters


class GaussianLogLikelihoodKnownSigma(BaseLikelihood):
    """
    This class represents a Gaussian Log Likelihood with a known sigma,
    which assumes that the data follows a Gaussian distribution and computes
    the log-likelihood of observed data under this assumption.

    Attributes:
        _logpi (float): Precomputed offset value for the log-likelihood function.
    """

    def __init__(self, problem: BaseProblem, sigma: List[float]):
        super(GaussianLogLikelihoodKnownSigma, self).__init__(problem, sigma)
        self.set_sigma(sigma)
        self._offset = -0.5 * self.n_time_data * np.log(2 * np.pi / self.sigma0)
        self._multip = -1 / (2.0 * self.sigma0**2)
        self.sigma2 = self.sigma0**-2
        self._dl = np.ones(self._n_parameters)

    def _evaluate(self, x: np.ndarray, grad: Union[None, np.ndarray] = None) -> float:
        """
        Evaluates the Gaussian log-likelihood for the given parameters with known sigma.
        """
        y = self.problem.evaluate(x)
        if any(
            len(y.get(key, [])) != len(self._target.get(key, [])) for key in self.signal
        ):
            return -np.inf  # prediction doesn't match target

        e = np.sum(
            [
                np.sum(
                    self._offset
                    + self._multip * np.sum((self._target[signal] - y[signal]) ** 2)
                )
                for signal in self.signal
            ]
        )

        return e if self.n_outputs != 1 else e.item()

    def _evaluateS1(self, x, grad=None):
        """
        Calls the problem.evaluateS1 method and calculates the log-likelihood and gradient.
        """
        y, dy = self.problem.evaluateS1(x)
        if any(
            len(y.get(key, [])) != len(self._target.get(key, [])) for key in self.signal
        ):
            return -np.inf, -self._dl * np.ones(self.n_parameters)

        r = np.array([self._target[signal] - y[signal] for signal in self.signal])
        likelihood = self._evaluate(x)
        dl = np.sum((self.sigma2 * np.sum((r * dy.T), axis=2)), axis=1)
        return likelihood, dl


class GaussianLogLikelihood(BaseLikelihood):
    """
    This class represents a Gaussian Log Likelihood, which assumes that the
    data follows a Gaussian distribution and computes the log-likelihood of
    observed data under this assumption.

    Attributes:
        _logpi (float): Precomputed offset value for the log-likelihood function.
    """

    def __init__(self, problem: BaseProblem, sigma0=0.001, x0=0.005):
        super(GaussianLogLikelihood, self).__init__(problem)
        self._logpi = -0.5 * self.n_time_data * np.log(2 * np.pi)
        self._dl = np.inf * np.ones(self._n_parameters + self.n_outputs)
        self._dsigma_scale = 1e2

        # Set the bounds for the sigma parameters
        self.lower_bound = max((x0 - 6 * sigma0), 1e-4)
        self.upper_bound = x0 + 6 * sigma0
        self._validate_and_correct_length(sigma0, x0)

    @property
    def dsigma_scale(self):
        """
        Scaling factor for the dsigma term in the gradient calculation.
        """
        return self._dsigma_scale

    @dsigma_scale.setter
    def dsigma_scale(self, new_value):
        if new_value < 0:
            raise ValueError("dsigma_scale must be non-negative")
        self._dsigma_scale = new_value

    def _validate_and_correct_length(self, sigma0, x0):
        """
        Validate and correct the length of sigma0 and x0 arrays.
        """
        expected_length = len(self._dl)

        self.sigma0 = np.pad(
            self.sigma0,
            (0, max(0, expected_length - len(self.sigma0))),
            constant_values=sigma0,
        )
        self.x0 = np.pad(
            self.x0, (0, max(0, expected_length - len(self.x0))), constant_values=x0
        )

        if len(self.bounds["upper"]) != expected_length:
            num_elements_to_add = expected_length - len(self.bounds["upper"])
            self.bounds["lower"].extend([self.lower_bound] * num_elements_to_add)
            self.bounds["upper"].extend([self.upper_bound] * num_elements_to_add)

    def _evaluate(self, x: np.ndarray, grad: Union[None, np.ndarray] = None) -> float:
        """
        Evaluates the Gaussian log-likelihood for the given parameters.

        Args:
            x (np.ndarray): The parameters for which to evaluate the log-likelihood.
                            The last `self.n_outputs` elements are assumed to be the
                            standard deviations of the Gaussian distributions.

        Returns:
            float: The log-likelihood value, or -inf if the standard deviations are non-positive.
        """
        sigma = np.asarray(x[-self.n_outputs :])
        if np.any(sigma <= 0):
            return -np.inf

        y = self.problem.evaluate(x[: -self.n_outputs])
        if any(
            len(y.get(key, [])) != len(self._target.get(key, [])) for key in self.signal
        ):
            return -np.inf  # prediction doesn't match target

        e = np.sum(
            [
                np.sum(
                    self._logpi
                    - self.n_time_data * np.log(sigma)
                    - np.sum((self._target[signal] - y[signal]) ** 2) / (2.0 * sigma**2)
                )
                for signal in self.signal
            ]
        )

        return e if self.n_outputs != 1 else e.item()

    def _evaluateS1(
        self, x: np.ndarray, grad: Union[None, np.ndarray] = None
    ) -> Tuple[float, np.ndarray]:
        """
        Calls the problem.evaluateS1 method and calculates the log-likelihood.

        Args:
            x (np.ndarray): The parameters for which to evaluate the log-likelihood.
            grad (Union[None, np.ndarray]): The gradient (optional).

        Returns:
            Tuple[float, np.ndarray]: The log-likelihood and its gradient.
        """
        sigma = np.asarray(x[-self.n_outputs :])
        if np.any(sigma <= 0):
            return -np.inf, -self._dl

        y, dy = self.problem.evaluateS1(x[: -self.n_outputs])
        if any(
            len(y.get(key, [])) != len(self._target.get(key, [])) for key in self.signal
        ):
            return -np.inf, -self._dl

        r = np.array([self._target[signal] - y[signal] for signal in self.signal])
        likelihood = self._evaluate(x)
        dl = np.sum((sigma ** (-2.0) * np.sum((r * dy.T), axis=2)), axis=1)
        dsigma = (
            -self.n_time_data / sigma + sigma ** (-3.0) * np.sum(r**2, axis=1)
        ) / self._dsigma_scale
        dl = np.concatenate((dl.flatten(), dsigma))

        return likelihood, dl
