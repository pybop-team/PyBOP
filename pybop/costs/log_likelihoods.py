import numpy as np

from pybop.costs.error_measures import ErrorMeasure
from pybop.parameters.distributions import Uniform
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
