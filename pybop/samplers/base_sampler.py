from dataclasses import dataclass

import numpy as np

from pybop.problems.base_problem import Problem


@dataclass
class SamplerOptions:
    """
    Base options for the sampler.

    Attributes:
        n_chains (int): The number of chains to concurrently sample from.
        x0 (float | np.ndarray): Initial guess for the parameter values.
        cov (float | np.ndarray): Covariance matrix.
    """

    n_chains: int = 1
    x0: float | np.ndarray | None = None
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
    problem : Problem
        The problem representing the negative unnormalised posterior distribution.
    options : SamplerOptions, optional
        Options for the sampler, by default SamplerOptions().
    """

    def __init__(
        self,
        problem: Problem,
        options: SamplerOptions | None = None,
    ):
        self._problem = problem
        self._options = options or SamplerOptions()
        self._options.validate()

        # Get initial conditions
        self._x0 = self.problem.params.get_initial_values(transformed=True) * np.ones(
            [self._options.n_chains, 1]
        )

        param_dims = len(self.problem.params)
        if np.isscalar(self._options.cov):
            self._cov0 = np.eye(param_dims) * self._options.cov
        else:
            self._cov0 = np.atleast_2d(self._options.cov)

    @staticmethod
    def default_options() -> SamplerOptions:
        """
        Get the default options for the sampler.

        Returns:
            SamplerOptions: Default options for the sampler.
        """
        return SamplerOptions()

    @property
    def x0(self) -> np.ndarray:
        return self._x0

    @property
    def cov0(self) -> np.ndarray:
        return self._cov0

    @property
    def problem(self) -> Problem:
        return self._problem

    @property
    def options(self) -> SamplerOptions:
        return self._options

    def run(self) -> np.ndarray:
        """
        Sample from the posterior distribution.

        Returns:
            np.ndarray: Samples from the posterior distribution.
        """
        raise NotImplementedError

    def set_initial_phase_iterations(self, iterations=250):
        """
        Set the number of iterations for the initial phase of the sampler.

        Args:
            iterations (int): Number of iterations for the initial phase.
        """
        self._initial_phase_iterations = iterations

    def set_max_iterations(self, iterations=500):
        """
        Set the maximum number of iterations for the sampler.

        Args:
            iterations (int): Maximum number of iterations.
        """
        iterations = int(iterations)
        if iterations < 1:
            raise ValueError("Number of iterations must be greater than 0")

        self._max_iterations = iterations
