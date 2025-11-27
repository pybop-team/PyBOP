from dataclasses import dataclass

import numpy as np

from pybop._result import SamplingResult
from pybop.problems.base_problem import Problem


@dataclass
class SamplerOptions:
    """
    Base options for the sampler.

    Attributes
    ----------
    n_chains : int
        The number of chains to concurrently sample from.
    cov : float | np.ndarray
        Covariance matrix.
    """

    n_chains: int = 1
    cov: float | np.ndarray = 0.05

    def validate(self):
        """
        Validate the options.

        Raises
        ------
        ValueError
            If the options are invalid.
        """
        if self.n_chains < 1:
            raise ValueError("Number of chains must be greater than 0.")


class BaseSampler:
    """
    Base class for Monte Carlo samplers.

    Parameters
    ----------
    log_pdf : pybop.Problem
        The negative unnormalised posterior distribution.
    options : SamplerOptions, optional
        Options for the sampler. If None, default options are used.
    """

    def __init__(
        self,
        log_pdf: Problem,
        options: SamplerOptions | None = None,
    ):
        self._log_pdf = log_pdf
        self._options = options or self.default_options()
        self._options.validate()

        # Get initial conditions
        self._x0 = self._log_pdf.parameters.get_initial_values(
            transformed=True
        ) * np.ones([self._options.n_chains, 1])

        param_dims = len(self._log_pdf.parameters)
        if np.isscalar(self._options.cov):
            self._cov0 = np.eye(param_dims) * self._options.cov
        else:
            self._cov0 = np.atleast_2d(self._options.cov)

    @staticmethod
    def default_options() -> SamplerOptions:
        """Get the default options for the sampler."""
        return SamplerOptions()

    @property
    def x0(self) -> np.ndarray:
        return self._x0

    @property
    def cov0(self) -> np.ndarray:
        return self._cov0

    @property
    def log_pdf(self) -> Problem:
        return self._log_pdf

    @property
    def options(self) -> SamplerOptions:
        return self._options

    def run(self) -> SamplingResult:
        """
        Sample from the posterior distribution.

        Returns:
            np.ndarray: Samples from the posterior distribution.
        """
        raise NotImplementedError

    def set_initial_phase_iterations(self, iterations: int = 250):
        """Set the number of iterations for the initial phase of the sampler."""
        self._initial_phase_iterations = iterations

    def set_max_iterations(self, iterations: int = 500):
        """Set the maximum number of iterations for the sampler."""
        iterations = int(iterations)
        if iterations < 1:
            raise ValueError("Number of iterations must be greater than 0.")

        self._max_iterations = iterations

    def set_warm_up_iterations(self, iterations: int = 250):
        """Set the number of warm up iterations for the sampler."""
        iterations = int(iterations)
        if iterations < 1:
            raise ValueError("Number of iterations must be greater than 0.")

        self._warm_up = iterations
