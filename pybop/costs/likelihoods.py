import numpy as np
import scipy.stats as stats

from pybop.costs.error_measures import ErrorMeasure
from pybop.parameters.distributions import Distribution, Uniform
from pybop.parameters.parameter import Inputs, Parameter, Parameters
from pybop.processing.dataset import Dataset


class LogLikelihood(ErrorMeasure):
    """
    Base class for likelihoods.

    Exists to distinguish between error measures and likelihood-based costs.
    """

    def __init__(self, dataset: Dataset, target: str | list[str] = None):
        super().__init__(dataset=dataset, target=target)
        self.minimising = False
        self.sigma = None
        self.parameters = Parameters()

    def set_sigma(self, sigma: np.ndarray | float, n_outputs: int, n_data: int):
        """Set the noise variance (sigma) after checking its validity."""
        raise NotImplementedError


class GaussianLogLikelihoodKnownSigma(LogLikelihood):
    """
    This class represents a Gaussian log-likelihood with a known sigma, which evaluates the
    log-likelihood under the assumption that measurement noise on the target data follows a
    Gaussian distribution.

    Parameters
    ----------
    sigma : scalar or array
        Initial standard deviation around ``x0``. Either a scalar value (one standard deviation
        for all coordinates) or an array with one entry per dimension.
    """

    def __init__(
        self,
        dataset: Dataset,
        sigma: list[float] | float,
        target: str | list[str] = None,
    ):
        super().__init__(dataset=dataset, target=target)
        self.set_sigma(sigma)

    def __call__(
        self,
        r: np.ndarray,
        dy: np.ndarray | None = None,
        inputs: Inputs | None = None,
    ) -> float | tuple[float, np.ndarray]:
        """
        Compute the Gaussian log-likelihood for the given parameters with known sigma.
        """
        l = np.sum(self._offset + self._multip * np.sum(np.real(r * np.conj(r))))

        if dy is not None:
            dl = {}
            for key, value in dy.items():
                dl[key] = -np.sum(np.sum((r * value), axis=1) / self.sigma2)
            return l, dl

        return l

    def set_sigma(self, sigma: np.ndarray | float):
        """Set sigma after checking its validity."""
        sigma = np.asarray(sigma, dtype=float)
        if not np.all(sigma > 0):
            raise ValueError("Sigma0 must be positive")
        if np.shape(sigma) not in [(), (1,), (self.n_outputs,)]:
            raise ValueError(
                "sigma must be either a scalar value (one standard deviation for "
                "all coordinates) or an array with one entry per dimension."
            )
        self.sigma2 = sigma**2.0
        self._offset = -0.5 * self.n_data * np.log(2 * np.pi * self.sigma2)
        self._multip = -1 / (2.0 * self.sigma2)


class GaussianLogLikelihood(LogLikelihood):
    """
    This class represents a Gaussian log-likelihood, which evaluates the log-likelihood under
    the assumption that measurement noise on the target data follows a Gaussian distribution.

    This class estimates the standard deviation of the Gaussian distribution alongside the
    parameters of the model.

    Attributes
    ----------
    _logpi : float
        Precomputed offset value for the log-likelihood function.
    """

    def __init__(
        self,
        dataset: Dataset,
        sigma: float | list[float] | list[Parameter] = 1e-2,
        target: str | list[str] = None,
    ):
        super().__init__(dataset=dataset, target=target)
        self.set_sigma(sigma)
        self._logpi = -0.5 * self.n_data * np.log(2 * np.pi)

    def set_sigma(self, sigma: float | list[float] | list[Parameter]):
        # Reset
        self.parameters = Parameters()
        self.sigma = Parameters()

        # Compile sigma parameters
        sigma = [sigma] if not isinstance(sigma, list) else sigma
        sigma = self._pad_sigma(sigma)

        for i, value in enumerate(sigma):
            self._add_single_sigma(i, value)

    def _pad_sigma(self, sigma):
        if len(sigma) < self.n_outputs:
            return np.pad(
                sigma,
                (0, self.n_outputs - len(sigma)),
                constant_values=sigma[-1],
            )
        return sigma

    def _add_single_sigma(self, index, value):
        if isinstance(value, Parameter):
            sigma = value
        elif isinstance(value, int | float):
            sigma = Parameter(
                distribution=Uniform(1e-8 * value, 3 * value),
                initial_value=value,
            )
        else:
            raise TypeError(
                f"Expected sigma to contain Parameter objects or numeric values. "
                f"Received {type(value)}"
            )
        self.sigma.add(f"Sigma for output {index + 1}", sigma)
        self.parameters.add(f"Sigma for output {index + 1}", sigma)

    def __call__(
        self,
        r: np.ndarray,
        dy: np.ndarray | None = None,
        inputs: Inputs | None = None,
    ) -> float | tuple[float, np.ndarray]:
        """
        Compute the Gaussian log-likelihood for the given parameters.
        """
        inputs = inputs or self.parameters.to_dict("initial")
        sigma_values = np.zeros(len(self.sigma))
        for i, name in enumerate(self.sigma.names):
            sigma_values[i] = inputs[name]

        sum_r2 = np.sum(np.real(r * np.conj(r)), axis=1)
        l = np.sum(
            self._logpi
            - self.n_data * np.log(sigma_values)
            - sum_r2 / (2.0 * sigma_values**2.0)
        )

        if dy is not None:
            dl = {}
            for key, value in dy.items():
                dl[key] = -np.sum(np.sum((r * value), axis=1) / (sigma_values**2.0))
            for i, (key, value) in enumerate(
                zip(self.sigma.names, sigma_values, strict=False)
            ):
                dl[key] = -self.n_data / value + sum_r2[i] / (value**3.0)
            return l, dl

        return l


class LogPosterior(LogLikelihood):
    """
    The log-posterior defined as the sum of the log-likelihood and the log-prior.

    Additional Parameters
    ---------------------
    log_likelihood : LogLikelihood
        The likelihood class of type ``LogLikelihood``.
    prior : Optional, Union[pybop.Parameter, stats.distributions.rv_frozen]
        The prior class of type ``Parameter``, ``Distribution`` or ``stats.distributions.rv_frozen``.
        If not provided, the prior class will be taken from the parameter distributions
        constructed in the `pybop.Parameters` class.
    """

    def __init__(
        self,
        log_likelihood: LogLikelihood,
        prior: Parameter | stats.distributions.rv_frozen | Distribution | None = None,
    ):
        dataset = Dataset(log_likelihood.dataset)
        dataset.domain = log_likelihood.domain
        super().__init__(dataset=dataset, target=log_likelihood.target)
        self.log_likelihood = log_likelihood
        self.parameters = self.log_likelihood.parameters
        self.prior = prior
        self.joint_prior = None  # must be built with model parameters included

    def set_joint_prior(self):
        if self.prior is None:
            self.joint_prior = self.parameters.distribution
        elif isinstance(self.prior, (stats.distributions.rv_frozen)):
            self.joint_prior = Distribution(self.prior)
        elif isinstance(self.prior, Parameter):
            self.joint_prior = self.prior.distribution
        elif isinstance(self.prior, Distribution):
            self.joint_prior = self.prior
        else:
            raise TypeError(
                "All priors must either be of type pybop.Parameter, pybop.Distribution or scipy.stats.distributions.rv_frozen"
            )

    def __call__(
        self,
        r: np.ndarray,
        dy: np.ndarray | None = None,
        inputs: Inputs | None = None,
    ) -> float | tuple[float, np.ndarray]:
        # Get the values of all input parameters
        inputs = inputs or self.parameters.to_dict("initial")
        input_values = np.asarray(list(inputs.values()))

        # Compute log prior (and gradient)
        if dy is not None:
            log_prior, dp = self.joint_prior.logpdfS1(input_values)
            dp = {key: dp[i] for i, key in enumerate(self.parameters.names)}
        else:
            log_prior = self.joint_prior.logpdf(input_values)

        if not np.isfinite(log_prior).any():
            return self.failure(self.parameters.names, dy)

        # Compute log likelihood and add log prior (and gradients)
        if dy is not None:
            log_likelihood, dl = self.log_likelihood(r, dy, inputs=inputs)

            dp = {key: dp[key] + dl[key] for i, key in enumerate(self.parameters.names)}
            return log_likelihood + log_prior, dp

        return self.log_likelihood(r, inputs=inputs) + log_prior
