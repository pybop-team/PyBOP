import logging
import time
from dataclasses import dataclass

import numpy as np
import pints

from pybop import (
    BaseSampler,
    MultiChainProcessor,
    SingleChainProcessor,
)
from pybop.problems.base_problem import Problem
from pybop.samplers.base_sampler import SamplerOptions


@dataclass
class PintsSamplerOptions(SamplerOptions):
    max_iterations: int = 500
    n_workers: int = 1
    chains_in_memory: bool = True
    log_to_screen: bool = True
    log_filename: str | None = None
    initial_phase_iterations: int = 250
    parallel: bool = False
    verbose: bool = False
    warm_up_iterations: int = 0
    chain_files: list[str] | None = None
    evaluation_files: list[str] | None = None

    def validate(self):
        """
        Validate the options.

        Raises
        ------
        ValueError
            If the options are invalid.
        """
        super().validate()
        if self.cov is not None and self.cov <= 0:
            raise ValueError("Sigma must be positive.")
        if self.warm_up_iterations < 0:
            raise ValueError("Number of warm-up steps must be non-negative.")
        if self.max_iterations < 1:
            raise ValueError("Maximum number of iterations must be greater than 0.")
        if self.n_workers < 1:
            raise ValueError("Number of workers must be greater than 0.")
        if self.initial_phase_iterations < 1:
            raise ValueError(
                "Number of initial phase iterations must be greater than 0."
            )


class BasePintsSampler(BaseSampler):
    """
    Base class for PINTS samplers.

    This class extends the BaseSampler class to provide a common interface for
    PINTS samplers. The class provides a sample() method that can be used to
    sample from the posterior distribution using a PINTS sampler.

    Parameters
    ----------
    problem: pybop.Problem
        The problem representing the negative unnormalised posterior distribution.
    sampler: pints.MCMCSampler
        The PINTS sampler to be used for sampling.
    options: Optional[PintsSamplerOptions]
        Options for the sampler, by default None.
    """

    def __init__(
        self,
        problem: Problem,
        sampler: type[pints.SingleChainMCMC | pints.MultiChainMCMC],
        options: PintsSamplerOptions | None = None,
    ):
        options = options or PintsSamplerOptions()
        super().__init__(problem, options=options)
        self._sampler = sampler
        self._max_iterations = options.max_iterations
        self._chains_in_memory = options.chains_in_memory
        self._log_to_screen = options.log_to_screen
        self._log_filename = options.log_filename
        self._initial_phase_iterations = options.initial_phase_iterations
        self._verbose = options.verbose
        self._warm_up = options.warm_up_iterations
        self._n_parameters = len(self.problem.params)
        self._chain_files = options.chain_files
        self._evaluation_files = options.evaluation_files
        self._loop_iters = 0
        self._iteration = 0
        self.iter_time = 0.0

        # Single chain vs multiple chain samplers
        self._single_chain = issubclass(self._sampler, pints.SingleChainMCMC)

        # Construct the samplers object
        if self._single_chain:
            self._n_samplers = self.options.n_chains
            self._samplers = [self._sampler(x0, sigma0=self.cov0) for x0 in self.x0]
        else:
            self._n_samplers = 1
            self._samplers = [self._sampler(self.options.n_chains, self.x0, self.cov0)]

        # Check for sensitivities from sampler and set evaluation
        self._needs_sensitivities = self._samplers[0].needs_sensitivities()

        # Check initial phase
        self._initial_phase = self._samplers[0].needs_initial_phase()
        if self._initial_phase:
            self.set_initial_phase_iterations()

    @staticmethod
    def default_options() -> PintsSamplerOptions:
        return PintsSamplerOptions()

    def _initialise_chain_processor(self):
        """
        Initialise the appropriate chain processor based on configuration.
        """
        if self._single_chain:
            self._chain_processor = SingleChainProcessor(self)
        else:
            self._chain_processor = MultiChainProcessor(self)

    def run(self) -> np.ndarray:
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
            return np.array([]).reshape((0, self._max_iterations, self._n_parameters))

        if self._warm_up > 0:
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
        # Construct function for evaluation
        if self._needs_sensitivities:

            def fun(x):
                self.problem.set_params(x)
                return self.problem.run_with_sensitivities()

        else:

            def fun(x):
                self.problem.set_params(x)
                return self.problem.run()

        # Handle parallel case
        if self.options.parallel:
            return pints.ParallelEvaluator(fun, n_workers=self.options.n_workers)

        return pints.SequentialEvaluator(fun)

    def _initialise_storage(self):
        # Storage of the received samples
        n_chains = self.options.n_chains
        self._sampled_logpdf = np.zeros(n_chains)
        self._sampled_prior = np.zeros(n_chains)

        # Pre-allocate arrays for chain storage
        storage_shape = (
            (n_chains, self._max_iterations, self._n_parameters)
            if self._chains_in_memory
            else (n_chains, self._n_parameters)
        )
        self._samples = np.zeros(storage_shape)

        self._evaluations = np.zeros((n_chains, self._max_iterations))

        # From PINTS:
        # Some samplers need intermediate steps, where `None` is returned instead
        # of a sample. But samplers can run asynchronously, so that one can return
        # `None` while another returns a sample. To deal with this, we maintain a
        # list of 'active' samplers that have not reached `max_iterations`,
        # and store the number of samples so far in each chain.
        if self._single_chain:
            self._active = list(range(n_chains))
            self._n_samples = [0] * n_chains

    def _initialise_logging(self):
        logging.basicConfig(format="%(message)s", level=logging.INFO)

        if self._log_to_screen:
            logging.info("Using " + str(self._samplers[0].name()))
            logging.info("Generating " + str(self.options.n_chains) + " chains.")
            if self.options.parallel:
                logging.info(
                    f"Running in parallel with {self.options.n_workers} worker processes."
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
