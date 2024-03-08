import numpy as np
from pybop.costs.base_cost import BaseCost


class BaseLikelihood(BaseCost):
    """
    Base class for likelihoods
    """

    def __init__(self, problem, sigma=None):
        super(BaseLikelihood, self).__init__(problem)
        self._n_output = problem.n_outputs
        self._n_times = problem.n_time_data
        self.sigma0 = sigma or np.zeros(self._n_output)
        self._n_parameters = problem.n_parameters
        self.log_likelihood = problem
        self._minimising = False

    def set_sigma(self, sigma):
        """
        Setter for sigma parameter
        """
        if np.any(sigma <= 0):
            raise ValueError("Sigma must not be negative")
        else:
            self.sigma0 = sigma

    def get_sigma(self):
        """
        Getter for sigma parameter
        """
        return self.sigma0

    def get_n_parameters(self):
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

    def __init__(self, problem, sigma=None):
        super(GaussianLogLikelihoodKnownSigma, self).__init__(problem, sigma=sigma)
        if sigma is not None:
            self.set_sigma(sigma)
        self._offset = -0.5 * self._n_times * np.log(2 * np.pi / self.sigma0)
        self._multip = -1 / (2.0 * self.sigma0**2)
        self.sigma2 = self.sigma0**-2
        self._dl = np.ones(self._n_parameters)

    def _evaluate(self, x, grad=None):
        """
        Calls the problem.evaluate method and calculates
        the log-likelihood
        """
        e = self._target - self.problem.evaluate(x)
        return np.sum(self._offset + self._multip * np.sum(e**2, axis=0))

    def _evaluateS1(self, x, grad=None):
        """
        Calls the problem.evaluateS1 method and calculates
        the log-likelihood
        """

        y, dy = self.problem.evaluateS1(x)
        if len(y) < len(self._target):
            likelihood = -np.float64(np.inf)
            dl = self._dl * np.ones(self._n_parameters)
        else:
            dy = dy.reshape(
                (
                    self._n_times,
                    self._n_output,
                    self._n_parameters,
                )
            )
            e = self._target - y
            likelihood = np.sum(self._offset + self._multip * np.sum(e**2, axis=0))
            dl = np.sum((self.sigma2 * np.sum((e.T * dy.T), axis=2)), axis=1)

        return likelihood, dl


class GaussianLogLikelihood(BaseLikelihood):
    """
    This class represents a Gaussian Log Likelihood, which assumes that the
    data follows a Gaussian distribution and computes the log-likelihood of
    observed data under this assumption.

    Attributes:
        _logpi (float): Precomputed offset value for the log-likelihood function.
    """

    def __init__(self, problem):
        super(GaussianLogLikelihood, self).__init__(problem)
        self._logpi = -0.5 * self._n_times * np.log(2 * np.pi)
        self._dl = np.ones(self._n_parameters + self._n_output)

    def _evaluate(self, x, grad=None):
        """
        Evaluates the Gaussian log-likelihood for the given parameters.

        Args:
            x (array_like): The parameters for which to evaluate the log-likelihood.
                             The last `self._n_output` elements are assumed to be the
                             standard deviations of the Gaussian distributions.

        Returns:
            float: The log-likelihood value, or -inf if the standard deviations are received as non-positive.
        """
        sigma = np.asarray(x[-self._n_output :])

        if np.any(sigma <= 0):
            return -np.inf

        e = self._target - self.problem.evaluate(x[: -self._n_output])
        return np.sum(
            self._logpi
            - self._n_times * np.log(sigma)
            - np.sum(e**2, axis=0) / (2.0 * sigma**2)
        )

    def _evaluateS1(self, x, grad=None):
        """
        Calls the problem.evaluateS1 method and calculates
        the log-likelihood
        """
        sigma = np.asarray(x[-self._n_output :])

        if np.any(sigma <= 0):
            return -np.inf, self._dl

        y, dy = self.problem.evaluateS1(x[: -self._n_output])
        if len(y) < len(self._target):
            likelihood = -np.float64(np.inf)
            dl = self._dl
        else:
            dy = dy.reshape(
                (
                    self._n_times,
                    self._n_output,
                    self._n_parameters,
                )
            )
            e = self._target - y
            likelihood = self._evaluate(x)
            dl = np.sum((sigma ** -(2.0) * np.sum((e.T * dy.T), axis=2)), axis=1)

            # Add sigma gradient to dl
            dsigma = -self._n_times / sigma + sigma ** -(3.0) * np.sum(e**2, axis=0)
            dl = np.concatenate((dl, dsigma))

        return likelihood, dl
