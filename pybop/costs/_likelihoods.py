import numpy as np
from pybop.costs.base_cost import BaseCost


class BaseLikelihood(BaseCost):
    """
    Base class for likelihoods
    """

    def __init__(self, problem, sigma=None):
        super(BaseLikelihood, self).__init__(problem, sigma)
        self._n_times = problem.n_time_data

    def set_sigma(self, sigma):
        """
        Setter for sigma parameter
        """

        if not isinstance(sigma, np.ndarray):
            sigma = np.array(sigma)

        if not np.issubdtype(sigma.dtype, np.number):
            raise ValueError("Sigma must contain only numeric values")

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

    def __init__(self, problem, sigma):
        super(GaussianLogLikelihoodKnownSigma, self).__init__(problem, sigma)
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
        prediction = self.problem.evaluate(x)

        for key in self.signal:
            if len(prediction.get(key, [])) != len(self._target.get(key, [])):
                return -np.float64(np.inf)  # prediction doesn't match target

        e = np.array(
            [
                np.sum(
                    self._offset
                    + self._multip
                    * np.sum((self._target[signal] - prediction[signal]) ** 2)
                )
                for signal in self.signal
            ]
        )

        if self.n_outputs == 1:
            return e.item()
        else:
            return np.sum(e)

    def _evaluateS1(self, x, grad=None):
        """
        Calls the problem.evaluateS1 method and calculates
        the log-likelihood
        """
        y, dy = self.problem.evaluateS1(x)
        for key in self.signal:
            if len(y.get(key, [])) != len(self._target.get(key, [])):
                likelihood = np.float64(np.inf)
                dl = self._dl * np.ones(self.n_parameters)
                return -likelihood, -dl

        r = np.array([self._target[signal] - y[signal] for signal in self.signal])
        likelihood = self._evaluate(x)

        if self.n_outputs == 1:
            r = r.reshape(self.problem.n_time_data)
            dy = dy.reshape(self.n_parameters, self.problem.n_time_data)
            dl = self.sigma2 * np.sum((r * dy), axis=1)
            return likelihood, dl
        else:
            r = r.reshape(self.n_outputs, self.problem.n_time_data)
            dl = self.sigma2 * np.sum((r[:, :, np.newaxis] * dy), axis=1)
            return likelihood, np.sum(dl, axis=1)


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
        self._dl = np.ones(self._n_parameters + self.n_outputs)

    def _evaluate(self, x, grad=None):
        """
        Evaluates the Gaussian log-likelihood for the given parameters.

        Args:
            x (array_like): The parameters for which to evaluate the log-likelihood.
                             The last `self.n_outputs` elements are assumed to be the
                             standard deviations of the Gaussian distributions.

        Returns:
            float: The log-likelihood value, or -inf if the standard deviations are received as non-positive.
        """
        sigma = np.asarray(x[-self.n_outputs :])

        if np.any(sigma <= 0):
            return -np.inf

        prediction = self.problem.evaluate(x[: -self.n_outputs])

        for key in self.signal:
            if len(prediction.get(key, [])) != len(self._target.get(key, [])):
                return -np.float64(np.inf)  # prediction doesn't match target

        e = np.array(
            [
                np.sum(
                    self._logpi
                    - self._n_times * np.log(sigma)
                    - np.sum((self._target[signal] - prediction[signal]) ** 2)
                    / (2.0 * sigma**2)
                )
                for signal in self.signal
            ]
        )

        if self.n_outputs == 1:
            return e.item()
        else:
            return np.sum(e)

    def _evaluateS1(self, x, grad=None):
        """
        Calls the problem.evaluateS1 method and calculates
        the log-likelihood
        """
        sigma = np.asarray(x[-self.n_outputs :])
        if np.any(sigma <= 0):
            return -np.float64(np.inf), -self._dl * np.ones(self.n_parameters)

        y, dy = self.problem.evaluateS1(x[: -self.n_outputs])
        for key in self.signal:
            if len(y.get(key, [])) != len(self._target.get(key, [])):
                likelihood = np.float64(np.inf)
                dl = self._dl * np.ones(self.n_parameters)
                return -likelihood, -dl

        r = np.array([self._target[signal] - y[signal] for signal in self.signal])
        likelihood = self._evaluate(x)

        if self.n_outputs == 1:
            r = r.reshape(self.problem.n_time_data)
            dy = dy.reshape(self.n_parameters, self.problem.n_time_data)
            dl = sigma ** (-2.0) * np.sum((r * dy), axis=1)
            dsigma = -self._n_times / sigma + sigma**-(3.0) * np.sum(r**2, axis=0)
            dl = np.concatenate((dl, dsigma))
            return likelihood, dl
        else:
            r = r.reshape(self.n_outputs, self.problem.n_time_data)
            dl = sigma ** (-2.0) * np.sum((r[:, :, np.newaxis] * dy), axis=1)
            dsigma = -self._n_times / sigma + sigma**-(3.0) * np.sum(r**2, axis=0)
            dl = np.concatenate((dl, dsigma))
            return likelihood, np.sum(dl, axis=1)
