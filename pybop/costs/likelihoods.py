import numpy as np
import scipy.stats as stats

from pybop._dataset import Dataset
from pybop.costs.error_measures import ErrorMeasure
from pybop.parameters.parameter import Parameter, Parameters
from pybop.parameters.priors import Distribution, JointPrior, Uniform


class LogLikelihood(ErrorMeasure):
    """
    Base class for likelihoods.

    Exists to distinguish between error measures and likelihood-based costs.
    """

    def __init__(self, dataset: Dataset, target: str | list[str] = None):
        super().__init__(dataset=dataset, target=target)
        self.minimising = False
        self.sigma0 = None
        self.parameters = Parameters()

    def set_sigma0(self, sigma0: np.ndarray | float, n_outputs: int, n_data: int):
        """Set sigma0 after checking its validity."""
        raise NotImplementedError


class GaussianLogLikelihoodKnownSigma(LogLikelihood):
    """
    This class represents a Gaussian log-likelihood with a known sigma, which computes the
    log-likelihood under the assumption that measurement noise on the target data follows a
    Gaussian distribution.

    Parameters
    ----------
    sigma0 : scalar or array
        Initial standard deviation around ``x0``. Either a scalar value (one standard deviation
        for all coordinates) or an array with one entry per dimension.
    """

    def __init__(
        self,
        dataset: Dataset,
        sigma0: list[float] | float,
        target: str | list[str] = None,
    ):
        super().__init__(dataset=dataset, target=target)
        self.set_sigma0(sigma0)

    def __call__(
        self,
        r: np.ndarray,
        dy: np.ndarray | None = None,
    ) -> float | tuple[float, np.ndarray]:
        """
        Compute the Gaussian log-likelihood for the given parameters with known sigma.
        """
        l = np.sum(self._offset + self._multip * np.sum(np.real(r * np.conj(r))))

        if dy is not None:
            dl = -np.sum((np.sum((r * dy), axis=2) / self.sigma2), axis=1)
            return l, dl

        return l

    def set_sigma0(self, sigma0: np.ndarray | float):
        """Set sigma0 after checking its validity."""
        sigma0 = np.asarray(sigma0, dtype=float)
        if not np.all(sigma0 > 0):
            raise ValueError("Sigma0 must be positive")
        if np.shape(sigma0) not in [(), (1,), (self.n_outputs,)]:
            raise ValueError(
                "sigma0 must be either a scalar value (one standard deviation for "
                "all coordinates) or an array with one entry per dimension."
            )
        self.sigma2 = sigma0**2.0
        self._offset = -0.5 * self.n_data * np.log(2 * np.pi * self.sigma2)
        self._multip = -1 / (2.0 * self.sigma2)


class GaussianLogLikelihood(LogLikelihood):
    """
    This class represents a Gaussian log-likelihood, which computes the log-likelihood under
    the assumption that measurement noise on the target data follows a Gaussian distribution.

    This class estimates the standard deviation of the Gaussian distribution alongside the
    parameters of the model.

    Attributes
    ----------
    _logpi : float
        Precomputed offset value for the log-likelihood function.
    _dsigma_scale : float
        Scale factor for derivative of standard deviation.
    """

    def __init__(
        self,
        dataset: Dataset,
        sigma0: float | list[float] | list[Parameter] = 1e-2,
        dsigma_scale: float = 1.0,
        target: str | list[str] = None,
    ):
        super().__init__(dataset=dataset, target=target)
        self.set_sigma0(sigma0)
        self._dsigma_scale = dsigma_scale
        self._logpi = -0.5 * self.n_data * np.log(2 * np.pi)

    def set_sigma0(self, sigma0: float | list[float] | list[Parameter]):
        # Reset
        self.parameters = Parameters()
        self.sigma = Parameters()

        # Compile sigma parameters
        sigma0 = [sigma0] if not isinstance(sigma0, list) else sigma0
        sigma0 = self._pad_sigma0(sigma0)

        for i, value in enumerate(sigma0):
            self._add_single_sigma(i, value)

    def _pad_sigma0(self, sigma0):
        if len(sigma0) < self.n_outputs:
            return np.pad(
                sigma0,
                (0, self.n_outputs - len(sigma0)),
                constant_values=sigma0[-1],
            )
        return sigma0

    def _add_single_sigma(self, index, value):
        if isinstance(value, Parameter):
            sigma = value
        elif isinstance(value, int | float):
            sigma = Parameter(
                f"Sigma for output {index + 1}",
                initial_value=value,
                prior=Uniform(1e-8 * value, 3 * value),
                bounds=[1e-8, 3 * value],
            )
        else:
            raise TypeError(
                f"Expected sigma0 to contain Parameter objects or numeric values. "
                f"Received {type(value)}"
            )
        self.sigma.add(sigma)
        self.parameters.add(sigma)

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

    def __call__(
        self,
        r: np.ndarray,
        dy: np.ndarray | None = None,
    ) -> float | tuple[float, np.ndarray]:
        """
        Compute the Gaussian log-likelihood for the given parameters.
        """
        sigma = self.sigma.get_values()

        sum_r2 = np.sum(np.real(r * np.conj(r)), axis=1)
        l = np.sum(
            self._logpi - self.n_data * np.log(sigma) - sum_r2 / (2.0 * sigma**2.0)
        )

        if dy is not None:
            dl = np.concatenate(
                (
                    -np.sum((np.sum((r * dy), axis=2) / (sigma**2.0)), axis=1),
                    (-self.n_data / sigma + sum_r2 / (sigma**3.0)) / self._dsigma_scale,
                )
            )
            return l, dl

        return l


class LogPosterior(LogLikelihood):
    """
    The log-posterior defined as the sum of the log-likelihood and the log-prior.

    Additional Parameters
    ---------------------
    log_likelihood : LogLikelihood
        The likelihood class of type ``LogLikelihood``.
    prior : Optional, Union[pybop.Distribution, stats.rv_continuous]
        The prior class of type ``Distribution`` or ``stats.rv_continuous``.
        If not provided, the prior class will be taken from the parameter priors
        constructed in the `pybop.Parameters` class.
    gradient_step : float, default: 1e-3
        The step size for the finite-difference gradient calculation
    """

    def __init__(
        self,
        log_likelihood: LogLikelihood,
        prior: Distribution | stats.rv_continuous | None = None,
        gradient_step: float = 1e-3,
    ):
        dataset = Dataset(log_likelihood.dataset)
        dataset.domain = log_likelihood.domain
        super().__init__(dataset=dataset, target=log_likelihood.target)
        self.log_likelihood = log_likelihood
        self.parameters = self.log_likelihood.parameters
        self.prior = prior
        self.joint_prior = None  # must be built with model parameters included
        self.gradient_step = gradient_step

    def set_joint_prior(self):
        if self.prior is None:
            self.joint_prior = JointPrior(*self.parameters.priors())
        elif isinstance(self.prior, (stats.rv_continuous | stats.distributions.rv_frozen)):
            self.joint_prior = Distribution(self.prior, gradient_step =self.gradient_step)
        else:
            self.joint_prior = self.prior

    def __call__(
        self,
        r: np.ndarray,
        dy: np.ndarray | None = None,
    ) -> float | tuple[float, np.ndarray]:
        # Compute log prior (and gradient)
        if dy is not None:
            log_prior, dp = self.joint_prior.logpdfS1(self.parameters.get_values())
        else:
            log_prior = self.joint_prior.logpdf(self.parameters.get_values())

        if not np.isfinite(log_prior).any():
            return self.failure(dy)

        # Compute log likelihood and add log prior (and gradients)
        if dy is not None:
            log_likelihood, dl = self.log_likelihood(r, dy)

            return log_likelihood + log_prior, dl + dp

        return self.log_likelihood(r) + log_prior
