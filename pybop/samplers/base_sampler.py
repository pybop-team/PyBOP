import logging
from typing import Union

import numpy as np
from pints import ParallelEvaluator

from pybop import LogPosterior


class BaseSampler:
    """
    Base class for Monte Carlo samplers.
    """

    def __init__(
        self, log_pdf: LogPosterior, x0, chains: int, cov0: Union[np.ndarray, float]
    ):
        """
        Initialise the base sampler.

        Parameters
        ----------------
        log_pdf (pybop.LogPosterior or List[pybop.LogPosterior]): The posterior or PDF to be sampled.
        chains (int): Number of chains to be used.
        x0: List-like initial condition for Monte Carlo sampling.
        cov0: The covariance matrix to be sampled.
        """
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

        # Check initial conditions
        if x0 is not None and len(x0) != self.n_parameters:
            raise ValueError("x0 must have the same number of parameters as log_pdf")

        # Set initial values, if x0 is None, initial values are unmodified.
        self.parameters.update(initial_values=x0 if x0 is not None else None)
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
        if parallel is True:
            self._parallel = True
            self._n_workers = ParallelEvaluator.cpu_count()
        elif parallel >= 1:
            self._parallel = True
            self._n_workers = int(parallel)
        else:
            self._parallel = False
            self._n_workers = 1

    def _ask_for_samples(self):
        if self._single_chain:
            return [self._samplers[i].ask() for i in self._active]
        else:
            return self._samplers[0].ask()

    def _check_initial_phase(self):
        # Set initial phase if needed
        if self._initial_phase:
            for sampler in self._samplers:
                sampler.set_initial_phase(True)

    def _end_initial_phase(self):
        for sampler in self._samplers:
            sampler.set_initial_phase(False)
        if self._log_to_screen:
            logging.info("Initial phase completed.")

    def _initialise_storage(self):
        self._prior = None
        if isinstance(self._log_pdf, LogPosterior):
            self._prior = self._log_pdf.prior

        # Storage of the received samples
        self._sampled_logpdf = np.zeros(self._n_chains)
        self._sampled_prior = np.zeros(self._n_chains)

        # Pre-allocate arrays for chain storage
        self._samples = np.zeros(
            (self._n_chains, self._max_iterations, self.n_parameters)
        )

        # Pre-allocate arrays for evaluation storage
        if self._prior:
            # Store posterior, likelihood, prior
            self._evaluations = np.zeros((self._n_chains, self._max_iterations, 3))
        else:
            # Store pdf
            self._evaluations = np.zeros((self._n_chains, self._max_iterations))

        # From PINTS:
        # Some samplers need intermediate steps, where `None` is returned instead
        # of a sample. But samplers can run asynchronously, so that one can return
        # `None` while another returns a sample. To deal with this, we maintain a
        # list of 'active' samplers that have not reached `max_iterations`,
        # and store the number of samples so far in each chain.
        if self._single_chain:
            self._active = list(range(self._n_chains))
            self._n_samples = [0] * self._n_chains

    def _initialise_logging(self):
        logging.basicConfig(format="%(message)s", level=logging.INFO)

        if self._log_to_screen:
            logging.info("Using " + str(self._samplers[0].name()))
            logging.info("Generating " + str(self._n_chains) + " chains.")
            if self._parallel:
                logging.info(
                    f"Running in parallel with {self._n_workers} worker processes."
                )
            else:
                logging.info("Running in sequential mode.")
            if self._chain_files:
                logging.info("Writing chains to " + self._chain_files[0] + " etc.")
            if self._evaluation_files:
                logging.info(
                    "Writing evaluations to " + self._evaluation_files[0] + " etc."
                )

    def _finalise_logging(self):
        if self._log_to_screen:
            logging.info(
                f"Halting: Maximum number of iterations ({self._iteration}) reached."
            )
