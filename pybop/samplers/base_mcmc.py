import logging
from typing import List, Optional, Union

import numpy as np
from pints import (
    MultiSequentialEvaluator,
    ParallelEvaluator,
    SequentialEvaluator,
    SingleChainMCMC,
)

from pybop import BaseCost, BaseSampler, LogPosterior


class BasePintsSampler(BaseSampler):
    """
    Base class for PINTS samplers.

    This class extends the BaseSampler class to provide a common interface for
    PINTS samplers. The class provides a sample() method that can be used to
    sample from the posterior distribution using a PINTS sampler.
    """

    def __init__(
        self,
        log_pdf: Union[BaseCost, List[BaseCost]],
        chains: int,
        sampler,
        burn_in=None,
        x0=None,
        cov0=None,
        transformation=None,
        **kwargs,
    ):
        """
        Initialise the base PINTS sampler.

        Args:
            log_pdf (pybop.BaseCost or List[pybop.BaseCost]): The cost distribution(s) to be sampled.
            chains (int): Number of chains to be used.
            sampler: The sampler class to be used.
            x0 (list): Initial states for the chains.
            cov0: Initial standard deviation for the chains.
            transformation: Transformation to be applied to the samples.
            kwargs: Additional keyword arguments.
        """
        super().__init__(x0, cov0)

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
        self.burn_in = burn_in
        self.n_parameters = (
            log_pdf[0].n_parameters
            if isinstance(log_pdf, list)
            else log_pdf.n_parameters
        )
        self._transformation = transformation

        # Check log_pdf
        if isinstance(log_pdf, BaseCost):
            self._multi_log_pdf = False
        else:
            if len(log_pdf) != chains:
                raise ValueError("Number of log pdf's must match number of chains")

            first_pdf_parameters = log_pdf[0].n_parameters
            for pdf in log_pdf:
                if not isinstance(pdf, BaseCost):
                    raise ValueError("All log pdf's must be instances of BaseCost")
                if pdf.n_parameters != first_pdf_parameters:
                    raise ValueError(
                        "All log pdf's must have the same number of parameters"
                    )

            self._multi_log_pdf = True

        # Transformations
        if transformation is not None:
            self._apply_transformation(transformation)

        self._log_pdf = log_pdf

        # Number of chains
        self._n_chains = chains
        if self._n_chains < 1:
            raise ValueError("Number of chains must be greater than 0")

        # Check initial conditions
        # len of x0 matching number of chains, number of parameters, etc.

        # Single chain vs multiple chain samplers
        self._single_chain = issubclass(sampler, SingleChainMCMC)

        # Construct the samplers object
        try:
            if self._single_chain:
                self._n_samplers = self._n_chains
                self._samplers = [sampler(x0, sigma0=self._cov0) for x0 in self._x0]
            else:
                self._n_samplers = 1
                self._samplers = [sampler(self._n_chains, self._x0, self._cov0)]
        except Exception as e:
            raise ValueError(f"Error constructing samplers: {e}")

        # Check for sensitivities from sampler and set evaluation
        self._needs_sensitivities = self._samplers[0].needs_sensitivities()

        # Check initial phase
        self._initial_phase = self._samplers[0].needs_initial_phase()
        if self._initial_phase:
            self.set_initial_phase_iterations()

        # Parallelisation (Might be able to move into parent class)
        self._n_workers = 1
        self.set_parallel(self._parallel)

    def _apply_transformation(self, transformation):
        # TODO: Implement transformation logic
        pass

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

        # Initialise iterations and evaluations
        self._iteration = 0
        self._evaluations = 0

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
            self._evaluations += len(self.fxs)

            if self._single_chain:
                self._process_single_chain()
                self._intermediate_step = min(self._n_samples) <= self._iteration
            else:
                self._process_multi_chain()

            if self._intermediate_step:
                continue

            self._iteration += 1
            if self._log_to_screen and self._verbose:
                logging.info(f"Iteration: {self._iteration}")  # TODO: Add more info

            if self._max_iterations and self._iteration >= self._max_iterations:
                running = False

        self._finalise_logging()

        if self.burn_in:
            self._samples = self._samples[:, self.burn_in :, :]

        return self._samples if self._chains_in_memory else None

    def _process_single_chain(self):
        self.fxs_iterator = iter(self.fxs)
        for i in list(self._active):
            reply = self._samplers[i].tell(next(self.fxs_iterator))
            if reply:
                y, fy, accepted = reply
                y_store = self._inverse_transform(y)
                if self._chains_in_memory:
                    self._samples[i][self._n_samples[i]] = y_store
                else:
                    self._samples[i] = y_store

                if accepted:
                    self._sampled_logpdf[i] = (
                        fy[0] if self._needs_sensitivities else fy
                    )  # Not storing sensitivities
                    if self._prior:
                        self._sampled_prior[i] = self._prior(y)

                e = self._sampled_logpdf[i]
                if self._prior:
                    e = [
                        e,
                        self._sampled_logpdf[i] - self._sampled_prior[i],
                        self._sampled_prior[i],
                    ]

                self._evaluations[i][self._n_samples[i]] = e
                self._n_samples[i] += 1
                if self._n_samples[i] == self._max_iterations:
                    self._active.remove(i)

    def _process_multi_chain(self):
        reply = self._samplers[0].tell(self.fxs)
        self._intermediate_step = reply is None
        if reply:
            ys, fys, accepted = reply
            ys_store = np.array([self._inverse_transform(y) for y in ys])
            if self._chains_in_memory:
                self._samples[:, self._iteration] = ys_store
            else:
                self._samples = ys_store

            es = []
            for i, y in enumerate(ys):
                if accepted[i]:
                    self._sampled_logpdf[i] = (
                        fys[0][i] if self._needs_sensitivities else fys[i]
                    )
                    if self._prior:
                        self._sampled_prior[i] = self._prior(ys[i])
                e = self._sampled_logpdf[i]
                if self._prior:
                    e = [
                        e,
                        self._sampled_logpdf[i] - self._sampled_prior[i],
                        self._sampled_prior[i],
                    ]
                es.append(e)

            for i, e in enumerate(es):
                self._evaluations[i, self._iteration] = e

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

    def _check_stopping_criteria(self):
        has_stopping_criterion = False
        has_stopping_criterion |= self._max_iterations is not None
        if not has_stopping_criterion:
            raise ValueError("At least one stopping criterion must be set.")

    def _create_evaluator(self):
        f = self._log_pdf
        # Check for sensitivities from sampler and set evaluator
        if self._needs_sensitivities:
            if not self._multi_log_pdf:
                f = f.evaluateS1
            else:
                f = [pdf.evaluateS1 for pdf in f]

        if self._parallel:
            if not self._multi_log_pdf:
                self._n_workers = min(self._n_workers, self._n_chains)
            return ParallelEvaluator(f, n_workers=self._n_workers)
        else:
            return (
                SequentialEvaluator(f)
                if not self._multi_log_pdf
                else MultiSequentialEvaluator(f)
            )

    def _check_initial_phase(self):
        # Set initial phase if needed
        if self._initial_phase:
            for sampler in self._samplers:
                sampler.set_initial_phase(True)

    def _inverse_transform(self, y):
        return self._transformation.to_model(y) if self._transformation else y

    def _initialise_storage(self):
        self._prior = None
        if isinstance(self._log_pdf, LogPosterior):
            self._prior = self._log_pdf.prior()

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

    def _end_initial_phase(self):
        for sampler in self._samplers:
            sampler.set_initial_phase(False)
        if self._log_to_screen:
            logging.info("Initial phase completed.")

    def _ask_for_samples(self):
        if self._single_chain:
            return [self._samplers[i].ask() for i in self._active]
        else:
            return self._samplers[0].ask()

    def _finalise_logging(self):
        if self._log_to_screen:
            logging.info(
                f"Halting: Maximum number of iterations ({self._iteration}) reached."
            )
