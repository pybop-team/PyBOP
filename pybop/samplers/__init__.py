import numpy as np
from pints import ParallelEvaluator
import warnings

class BaseSampler:
    """
    Base class for Monte Carlo samplers.
    """
    def __init__(self, x0, cov0):
        """
        Initialise the base sampler.

        Args:
            cost (pybop.cost): The cost to be sampled.
        """
        self._x0 = x0
        self._cov0 = cov0

    def run(self) -> np.ndarray:
        """
        Sample from the posterior distribution.

        Args:
            n_samples (int): Number of samples to draw.

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

    def set_parallel(self, parallel=False):
        """
        Enable or disable parallel evaluation.
        Credit: PINTS

        Parameters
        ----------
        parallel : bool or int, optional
            If True, use as many worker processes as there are CPU cores. If an integer, use that many workers.
            If False or 0, disable parallelism (default: False).
        """
        if parallel is True:
            self._parallel = True
            self._n_workers = ParallelEvaluator.cpu_count()
        elif parallel >= 1:
            self._parallel = True
            self._n_workers = int(parallel)
        else:
            self._parallel = False
            self._n_workers = 1
