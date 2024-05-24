import numpy as np

from pybop.costs.base_cost import BaseCost


class BaseLikelihood(BaseCost):
    """
    Base class for likelihoods
    """

    def __init__(self, problem, sigma=None):
        super(BaseLikelihood, self).__init__(problem, sigma)
        self.n_time_data = problem.n_time_data

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

    @property
    def n_parameters(self):
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
        self._offset = -0.5 * self.n_time_data * np.log(2 * np.pi / self.sigma0)
        self._multip = -1 / (2.0 * self.sigma0**2)
        self.sigma2 = self.sigma0**-2
        self._dl = np.ones(self._n_parameters)

    def _evaluate(self, x, grad=None):
        """
        Calls the problem.evaluate method and calculates
        the log-likelihood
        """
        y = self.problem.evaluate(x)

        for key in self.signal:
            if len(y.get(key, [])) != len(self._target.get(key, [])):
                return -np.float64(np.inf)  # prediction doesn't match target

        e = np.array(
            [
                np.sum(
                    self._offset
                    + self._multip * np.sum((self._target[signal] - y[signal]) ** 2)
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

    def __init__(self, problem):
        super(GaussianLogLikelihood, self).__init__(problem)
        self._logpi = -0.5 * self.n_time_data * np.log(2 * np.pi)
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

        y = self.problem.evaluate(x[: -self.n_outputs])

        for key in self.signal:
            if len(y.get(key, [])) != len(self._target.get(key, [])):
                return -np.float64(np.inf)  # prediction doesn't match target

        e = np.array(
            [
                np.sum(
                    self._logpi
                    - self.n_time_data * np.log(sigma)
                    - np.sum((self._target[signal] - y[signal]) ** 2) / (2.0 * sigma**2)
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
        dl = sigma ** (-2.0) * np.sum((r * dy.T), axis=2)
        dsigma = -self.n_time_data / sigma + sigma**-(3.0) * np.sum(r**2, axis=1)
        dl = np.concatenate((dl.flatten(), dsigma))
        return likelihood, dl


class LogPosterior(BaseCost):
    """
    The Log Posterior for a given problem.

    Computes the log posterior which is the sum of the log
    likelihood and the log prior.

    Inherits all parameters and attributes from ``BaseCost``.
    """

    def __init__(self, log_likelihood, log_prior=None, sigma=None):
        super(LogPosterior, self).__init__(problem=log_likelihood.problem, sigma=sigma)
        if self.sigma0 is None:
            self.sigma0 = []
            for param in self.problem.parameters:  # Update for parameters class
                self.sigma0.append(param.prior.sigma)

        # Store the likelihood and prior
        self._log_likelihood = log_likelihood
        self._prior = log_prior
        if self._prior is None:
            try:
                self._prior = [
                    param.prior
                    for i, param in enumerate(
                        log_likelihood.problem.parameters
                    )  # Update for parameters class
                ]
            except Exception as e:
                raise ValueError(
                    f"An error occurred when constructing the Prior class: {e}"
                )

        try:  # This is a patch, the n_parameters val needs to be updated across the codebase
            self._n_parameters = self._prior.n_parameters
        except AttributeError:
            self._n_parameters = len(self._prior)

    def _evaluate(self, x, grad=None):
        """
        Calculate the posterior cost for a given set of parameters.

        Parameters
        ----------
        x : array-like
            The parameters for which to evaluate the cost.
        grad : array-like, optional
            An array to store the gradient of the cost function with respect
            to the parameters.

        Returns
        -------
        float
            The posterior cost.
        """
        prior = self._prior(x)
        if prior == np.inf:
            return prior
        return prior + self._log_likelihood.evaluate(x)

    def _evaluateS1(self, x):
        """
        Compute the posterior with respect to the parameters.
        The method passes the likelihood gradient to the optimiser without modification.

        Parameters
        ----------
        x : array-like
            The parameters for which to compute the cost and gradient.

        Returns
        -------
        tuple
            A tuple containing the cost and the gradient. The cost is a float,
            and the gradient is an array-like of the same length as `x`.

        Raises
        ------
        ValueError
            If an error occurs during the calculation of the cost or gradient.
        """
        prior, dp = self._prior.evaluateS1(x)
        if prior == np.inf:
            return prior
        likelihood, dl = self._log_likelihood.evaluateS1(x)
        return prior + likelihood, dp + dl

    def prior(self):
        """
        Return the prior object.

        Returns
        -------
        object
            The prior object.
        """
        return self._prior

    def likelihood(self):
        """
        Returns the likelihood.

        Returns
        -------
        object
            The likelihood object.
        """
        return self._log_likelihood
