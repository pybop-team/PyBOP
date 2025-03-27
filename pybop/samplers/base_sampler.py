from typing import Union

import numpy as np
from pints import ParallelEvaluator

from pybop import CostInterface, LogPosterior


class BaseSampler(CostInterface):
    """
    Base class for Monte Carlo samplers.

    Parameters
    ----------
    log_pdf : pybop.LogPosterior or List[pybop.LogPosterior]
        The posterior or PDF to be sampled.
    chains : int
        Number of chains to be used.
    x0
        List-like initial values of the parameters for Monte Carlo sampling.
    cov0
        The covariance matrix to be sampled.

    Note: Samplers perform maximisation of the Posterior by default.
    """

    def __init__(
        self,
        log_pdf: Union[LogPosterior, list[LogPosterior]],
        x0,
        chains: int,
        cov0: Union[np.ndarray, float],
    ):
        self._log_pdf = log_pdf
        self._cov0 = cov0

        # Number of chains
        self._n_chains = chains
        if self._n_chains < 1:
            raise ValueError("Number of chains must be greater than 0")

        # Set up parameters based on log_pdf
        if isinstance(log_pdf, LogPosterior):
            self.parameters = log_pdf.parameters
            self.n_parameters = log_pdf.n_parameters
        elif isinstance(log_pdf, (list, np.ndarray)) and isinstance(
            log_pdf[0], LogPosterior
        ):
            self.parameters = log_pdf[0].parameters
            self.n_parameters = log_pdf[0].n_parameters
        else:
            raise ValueError(
                "log_pdf must be a LogPosterior or List[LogPosterior]"
            )  # TODO: Update for more general sampling

        transformation = self.parameters.construct_transformation()
        super().__init__(transformation=transformation)

        # Check initial conditions
        if x0 is not None:
            if len(x0) != self.n_parameters:
                raise ValueError(
                    "x0 must have the same number of parameters as log_pdf"
                )
            self.parameters.update(initial_values=x0)

        # Update x0 w/ transformation if applicable - reshape to align with chains
        self._x0 = self.parameters.reset_initial_value(apply_transform=True).reshape(
            1, -1
        )

        if len(self._x0) != self._n_chains or len(self._x0) == 1:
            self._x0 = np.tile(self._x0, (self._n_chains, 1))

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
        self._parallel = bool(parallel is True or parallel >= 1)

        if parallel is True:
            self._n_workers = ParallelEvaluator.cpu_count()
        elif parallel >= 1:
            self._n_workers = int(parallel)
        else:
            self._n_workers = 1
