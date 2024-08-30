import logging
from functools import partial
from typing import Optional

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
        log_pdf: LogPosterior,
        sampler,
        chains: int = 1,
        warm_up=None,
        x0=None,
        cov0=0.1,
        **kwargs,
    ):
        """
        Initialise the base PINTS sampler.

        Args:
            log_pdf (pybop.LogPosterior or List[pybop.LogPosterior]): The distribution(s) to be sampled.
            chains (int): Number of chains to be used.
            sampler: The sampler class to be used.
            x0 (list): Initial states for the chains.
            cov0: Initial standard deviation for the chains.
            kwargs: Additional keyword arguments.
        """
        super().__init__(log_pdf, x0, cov0)

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
        self._iteration = 0
        self._warm_up = warm_up
        self.n_parameters = (
            self._log_pdf[0].n_parameters
            if isinstance(self._log_pdf, list)
            else self._log_pdf.n_parameters
        )

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

        # Number of chains
        self._n_chains = chains
        if self._n_chains < 1:
            raise ValueError("Number of chains must be greater than 0")

        # Check initial conditions
        if self._x0.size != self.n_parameters:
            raise ValueError("x0 must have the same number of parameters as log_pdf")
        if len(self._x0) != self._n_chains or len(self._x0) == 1:
            self._x0 = np.tile(self._x0, (self._n_chains, 1))

        # Single chain vs multiple chain samplers
        self._single_chain = issubclass(sampler, SingleChainMCMC)

        # Construct the samplers object
        if self._single_chain:
            self._n_samplers = self._n_chains
            self._samplers = [sampler(x0, sigma0=self._cov0) for x0 in self._x0]
        else:
            self._n_samplers = 1
            self._samplers = [sampler(self._n_chains, self._x0, self._cov0)]

        # Check for sensitivities from sampler and set evaluation
        self._needs_sensitivities = self._samplers[0].needs_sensitivities()

        # Check initial phase
        self._initial_phase = self._samplers[0].needs_initial_phase()
        if self._initial_phase:
            self.set_initial_phase_iterations()

        # Parallelisation (Might be able to move into parent class)
        self._n_workers = 1
        self.set_parallel(self._parallel)

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

        if self._warm_up:
            self._samples = self._samples[:, self._warm_up :, :]

        return self._samples if self._chains_in_memory else None

    def _process_single_chain(self):
        self.fxs_iterator = iter(self.fxs)
        for i in list(self._active):
            reply = self._samplers[i].tell(next(self.fxs_iterator))
            if reply:
                y, fy, accepted = reply
                y_store = self._inverse_transform(
                    y, self._log_pdf[i] if self._multi_log_pdf else self._log_pdf
                )
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
            ys_store = np.asarray(
                [self._inverse_transform(y, self._log_pdf) for y in ys]
            )
            if self._chains_in_memory:
                self._samples[:, self._iteration] = ys_store
            else:
                self._samples = ys_store

            es = []
            for i, _y in enumerate(ys):
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
                f = partial(f, calculate_grad=True)
            else:
                f = [partial(pdf, calculate_grad=True) for pdf in f]

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

    def _inverse_transform(self, y, log_pdf):
        return log_pdf.transformation.to_model(y) if log_pdf.transformation else y
