import logging
import time
from functools import partial
from typing import Optional, Union

import numpy as np
from pints import (
    MultiSequentialEvaluator,
    ParallelEvaluator,
    SequentialEvaluator,
    SingleChainMCMC,
)

from pybop import (
    BaseCost,
    BaseSampler,
    LogPosterior,
    MultiChainProcessor,
    SingleChainProcessor,
)


class BasePintsSampler(BaseSampler):
    """
    Base class for PINTS samplers.

    This class extends the BaseSampler class to provide a common interface for
    PINTS samplers. The class provides a sample() method that can be used to
    sample from the posterior distribution using a PINTS sampler.

    Parameters
    ----------
    log_pdf : pybop.LogPosterior or List[pybop.LogPosterior]
        An object to be sampled, currently supports the pybop.LogPosterior class.
    sampler : pybop.Sampler
        The sampling algorithm to use.
    chains : int
        Number of chains to run concurrently. Each chain contains separate markov samples from
        the log_pdf.
    x0 : numpy.ndarray
        Initial values of the parameters for the optimisation.
    cov0 : list-like
        Initial covariance for the chains in the parameters. Either a scalar value
        (same for all coordinates) or an array with one entry per dimension.
    kwargs
        Additional keyword arguments.
    """

    def __init__(
        self,
        log_pdf: Union[LogPosterior, list[LogPosterior]],
        sampler,
        chains: int = 1,
        warm_up=None,
        x0=None,
        cov0=0.1,
        **kwargs,
    ):
        super().__init__(log_pdf, x0, chains, cov0)

        # Set kwargs
        self._max_iterations = kwargs.get("max_iterations", 500)
        self._log_to_screen = kwargs.get("log_to_screen", True)
        self._log_filename = kwargs.get("log_filename", None)
        self._initial_phase_iterations = kwargs.get("initial_phase_iterations", 250)
        self._chains_in_memory = kwargs.get("chains_in_memory", True)
        self._chain_files = kwargs.get("chain_files", None)
        self._evaluation_files = kwargs.get("evaluation_files", None)
        self._parallel = kwargs.get("parallel", False)
        self._verbose = kwargs.get("verbose", False)
        self.sampler = sampler
        self._prior = None
        self.iter_time = float(0)
        self._iteration = 0
        self._loop_iters = 0
        self._warm_up = warm_up

        # Check log_pdf
        if isinstance(self._log_pdf, BaseCost):
            self._multi_log_pdf = False
        else:
            if len(self._log_pdf) != chains:
                raise ValueError("Number of log pdf's must match number of chains")

            first_pdf_parameters = self._log_pdf[0].n_parameters
            for pdf in self._log_pdf:
                if not isinstance(pdf, BaseCost):
                    raise ValueError("All log pdf's must be instances of BaseCost")
                if pdf.n_parameters != first_pdf_parameters:
                    raise ValueError(
                        "All log pdf's must have the same number of parameters"
                    )

            self._multi_log_pdf = True

        # Single chain vs multiple chain samplers
        self._single_chain = issubclass(self.sampler, SingleChainMCMC)

        # Construct the samplers object
        if self._single_chain:
            self._n_samplers = self._n_chains
            self._samplers = [self.sampler(x0, sigma0=self._cov0) for x0 in self._x0]
        else:
            self._n_samplers = 1
            self._samplers = [self.sampler(self._n_chains, self._x0, self._cov0)]

        # Check for sensitivities from sampler and set evaluation
        self._needs_sensitivities = self._samplers[0].needs_sensitivities()

        # Check initial phase
        self._initial_phase = self._samplers[0].needs_initial_phase()
        if self._initial_phase:
            self.set_initial_phase_iterations()

        # Set parallelisation
        self.set_parallel(self._parallel)

    def _initialise_chain_processor(self):
        """
        Initialise the appropriate chain processor based on configuration.
        """
        if self._single_chain:
            self._chain_processor = SingleChainProcessor(self)
        else:
            self._chain_processor = MultiChainProcessor(self)

    def run(self) -> Optional[np.ndarray]:
        """
        Executes the Monte Carlo sampling process and generates samples
        from the posterior distribution.

        This method orchestrates the entire sampling process, managing
        iterations, evaluations, logging, and stopping criteria. It
        initialises the necessary structures, handles both single and
        multi-chain scenarios, and manages parallel or sequential
        evaluation based on the configuration.

        Returns:
            np.ndarray: A numpy array containing the samples from the
            posterior distribution if chains are stored in memory,
            otherwise returns None.

        Raises:
            ValueError: If no stopping criterion is set (i.e.,
            _max_iterations is None).

        Details:
            - Checks and ensures at least one stopping criterion is set.
            - Initialises iterations, evaluations, and other required
            structures.
            - Sets up the evaluator (parallel or sequential) based on the
            configuration.
            - Handles the initial phase, if applicable, and manages
            intermediate steps in the sampling process.
            - Logs progress and relevant information based on the logging
            configuration.
            - Iterates through the sampling process, evaluating the log
            PDF, updating chains, and managing the stopping criteria.
            - Finalises and returns the collected samples, or None if
            chains are not stored in memory.
        """

        self._initialise_logging()
        self._check_stopping_criteria()
        self._initialise_chain_processor()

        # Initialise iterations and evaluations
        self._iteration = 0

        evaluator = self._create_evaluator()
        self._check_initial_phase()
        self._initialise_storage()

        running = True
        while running:
            if (
                self._initial_phase
                and self._iteration == self._initial_phase_iterations
            ):
                self._end_initial_phase()

            xs = self._ask_for_samples()
            self.fxs = evaluator.evaluate(xs)
            self._process_chains()

            if self._single_chain:
                self._intermediate_step = min(self._n_samples) <= self._iteration

            # Skip the remaining loop logic
            if self._intermediate_step:
                continue

            self._iteration += 1
            if self._log_to_screen and self._verbose:
                if self._iteration <= 10 or self._iteration % 50 == 0:
                    timing_iterations = self._iteration - self._loop_iters
                    elapsed_time = time.time() - self.iter_time
                    iterations_per_second = (
                        timing_iterations / elapsed_time if elapsed_time > 0 else 0
                    )
                    logging.info(
                        f"| Iteration: {self._iteration} | Iter/s: {iterations_per_second: .2f} |"
                    )
                    self.iter_time = time.time()
                    self._loop_iters = self._iteration
            if self._max_iterations and self._iteration >= self._max_iterations:
                running = False

        self._finalise_logging()

        if not self._chains_in_memory:
            return None

        if self._warm_up:
            self._samples = self._samples[:, self._warm_up :, :]

        return self._samples

    def _process_chains(self):
        """
        Process chains using the appropriate processor.
        """
        self._chain_processor.process_chain()

    def _ask_for_samples(self):
        if self._single_chain:
            return [self._samplers[i].ask() for i in self._active]

        return self._samplers[0].ask()

    def _check_initial_phase(self):
        """
        Set initial phase if needed
        """
        if self._initial_phase:
            for sampler in self._samplers:
                sampler.set_initial_phase(True)

    def _end_initial_phase(self):
        for sampler in self._samplers:
            sampler.set_initial_phase(False)
        if self._log_to_screen:
            logging.info("Initial phase completed.")

    def _check_stopping_criteria(self):
        """
        Verify that at least one stopping criterion is defined.
        """
        if self._max_iterations is None:
            raise ValueError("At least one stopping criterion must be set.")

    def _create_evaluator(self):
        """
        Create appropriate evaluator based on configuration settings.
        """
        common_args = {"calculate_grad": self._needs_sensitivities}

        # Construct function for evaluation
        if not self._multi_log_pdf:
            f = partial(self.call_cost, cost=self._log_pdf, **common_args)
        else:
            f = [
                partial(self.call_cost, cost=log_pdf, **common_args)
                for log_pdf in self._log_pdf
            ]

        # Handle parallel case
        if self._parallel:
            # Adjust workers for single log pdf case
            if not self._multi_log_pdf:
                self._n_workers = min(self._n_workers, self._n_chains)
            return ParallelEvaluator(f, n_workers=self._n_workers)

        # Construct a dict for various return types
        evaluator_map = {False: SequentialEvaluator, True: MultiSequentialEvaluator}
        return evaluator_map[self._multi_log_pdf](f)

    def _initialise_storage(self):
        if isinstance(self._log_pdf, LogPosterior):
            self._prior = self._log_pdf.prior

        # Storage of the received samples
        self._sampled_logpdf = np.zeros(self._n_chains)
        self._sampled_prior = np.zeros(self._n_chains)

        # Pre-allocate arrays for chain storage
        storage_shape = (
            (self._n_chains, self._max_iterations, self.n_parameters)
            if self._chains_in_memory
            else (self._n_chains, self.n_parameters)
        )
        self._samples = np.zeros(storage_shape)

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

    @property
    def prior(self):
        return self._prior

    @property
    def samplers(self):
        return self._samplers

    @property
    def active(self):
        return self._active

    @property
    def single_chain(self):
        return self._single_chain

    @property
    def sampled_logpdf(self):
        return self._sampled_logpdf

    @property
    def sampled_prior(self):
        return self._sampled_prior

    @property
    def iteration(self):
        return self._iteration

    @property
    def needs_sensitivities(self):
        return self._needs_sensitivities

    @property
    def chains_in_memory(self):
        return self._chains_in_memory

    @property
    def n_samples(self):
        return self._n_samples

    @property
    def max_iterations(self):
        return self._max_iterations
