from time import time

import numpy as np
from pints import PSO as PintsPSO
from pints import NelderMead as PintsNelderMead
from pints import Optimiser as PintsOptimiser
from pints import ParallelEvaluator as PintsParallelEvaluator
from pints import PopulationBasedOptimiser
from pints import PopulationBasedOptimiser as PintsPopulationBasedOptimiser
from pints import RectangularBoundaries as PintsRectangularBoundaries
from pints import SequentialEvaluator as PintsSequentialEvaluator
from pints import strfloat as PintsStrFloat

from pybop import (
    AdamWImpl,
    BaseJaxCost,
    BaseOptimiser,
    GradientDescentImpl,
    OptimisationResult,
    SequentialJaxEvaluator,
)


class BasePintsOptimiser(BaseOptimiser):
    """
    A base class for defining optimisation methods from the PINTS library.

    Parameters
    ----------
    cost : callable
        The cost function to be minimized.
    pints_optimiser : class
        The PINTS optimiser class to be used.
    max_iterations : int, optional
        Maximum number of iterations for the optimisation.
    min_iterations : int, optional (default=2)
        Minimum number of iterations before termination.
    max_unchanged_iterations : int, optional (default=15)
        Maximum number of iterations without improvement before termination.
    parallel : bool, optional (default=False)
        Whether to run the optimisation in parallel.
    **optimiser_kwargs : optional
        Valid PINTS option keys and their values, for example:
        x0 : array_like
            Initial position from which optimization will start.
        sigma0 : float
            Initial step size or standard deviation depending on the optimiser.
        bounds : dict
            A dictionary with 'lower' and 'upper' keys containing arrays for lower and
            upper bounds on the parameters.
        use_f_guessed : bool
            Whether to track guessed function values.
        absolute_tolerance : float
            Absolute tolerance for convergence checking.
        relative_tolerance : float
            Relative tolerance for convergence checking.
        max_evaluations : int
            Maximum number of function evaluations.
        threshold : float
            Threshold value for early termination.
    """

    def __init__(
        self,
        cost,
        pints_optimiser,
        max_iterations: int = None,
        min_iterations: int = 2,
        max_unchanged_iterations: int = 15,
        multistart: int = 1,
        parallel: bool = False,
        **optimiser_kwargs,
    ):
        # First set attributes to default values
        self._boundaries = None
        self._needs_sensitivities = None
        self._use_f_guessed = None
        self._n_workers = 1
        self._callback = None
        self.set_parallel(parallel)
        self.set_max_iterations(max_iterations)
        self.set_min_iterations(min_iterations)
        self._unchanged_max_iterations = max_unchanged_iterations
        self._absolute_tolerance = 1e-5
        self._relative_tolerance = 1e-2
        self._max_evaluations = None
        self._threshold = None
        self._evaluations = None
        self._iterations = None
        self.option_methods = {
            "use_f_guessed": self.set_f_guessed_tracking,
            "max_evaluations": self.set_max_evaluations,
            "threshold": self.set_threshold,
        }

        self._pints_optimiser = pints_optimiser
        optimiser_kwargs["multistart"] = multistart
        super().__init__(cost, **optimiser_kwargs)

    def _set_up_optimiser(self):
        """
        Parse optimiser options and create an instance of the PINTS optimiser.
        """
        # Check and remove any duplicate keywords in self.unset_options
        self._sanitise_inputs()

        # Create an instance of the PINTS optimiser class
        if issubclass(self._pints_optimiser, PintsOptimiser):
            self.optimiser = self._pints_optimiser(
                self.x0, sigma0=self.sigma0, boundaries=self._boundaries
            )
        else:
            raise ValueError("The optimiser is not a recognised PINTS optimiser class.")

        # Check if sensitivities are required
        self._needs_sensitivities = self.optimiser.needs_sensitivities()

        # Apply additional options and remove them from options
        max_unchanged_kwargs = {"iterations": self._unchanged_max_iterations}
        for key, method in self.option_methods.items():
            if key in self.unset_options:
                method(self.unset_options.pop(key))

        # Capture tolerance options
        for tol_key in ["absolute_tolerance", "relative_tolerance"]:
            if tol_key in self.unset_options:
                max_unchanged_kwargs[tol_key] = self.unset_options.pop(tol_key)

        # Set population size (if applicable)
        if "population_size" in self.unset_options:
            population_size = self.unset_options.pop("population_size")
            self.set_population_size(population_size)

    def _sanitise_inputs(self):
        """
        Check and remove any duplicate optimiser options.
        """
        # Unpack values from any nested options dictionary
        if "options" in self.unset_options.keys():
            key_list = list(self.unset_options["options"].keys())
            for key in key_list:
                if key not in self.unset_options.keys():
                    self.unset_options[key] = self.unset_options["options"].pop(key)
                else:
                    raise Exception(
                        f"A duplicate {key} option was found in the options dictionary."
                    )
            self.unset_options.pop("options")

        # Convert bounds to PINTS boundaries
        if self.bounds is not None:
            ignored_optimisers = (GradientDescentImpl, AdamWImpl, PintsNelderMead)
            if issubclass(self._pints_optimiser, ignored_optimisers):
                print(f"NOTE: Boundaries ignored by {self._pints_optimiser}")
                self.bounds = None
            else:
                if issubclass(self._pints_optimiser, PintsPSO):
                    if not all(
                        np.isfinite(value)
                        for sublist in self.bounds.values()
                        for value in sublist
                    ):
                        raise ValueError(
                            f"Either all bounds or no bounds must be set for {self._pints_optimiser.__name__}."
                        )
                self._boundaries = PintsRectangularBoundaries(
                    self.bounds["lower"], self.bounds["upper"]
                )

    def name(self):
        """Returns the name of the PINTS optimisation strategy."""
        return self.optimiser.name()

    def _run(self):
        """
        Internal method to run the optimization using a PINTS optimiser.

        Returns
        -------
        result : pybop.Result
            The result of the optimisation including the optimised parameter values and cost.

        See Also
        --------
        This method is heavily based on the run method in the PINTS.OptimisationController class.
        """
        # Timing
        start_time = time()

        # Check stopping criteria
        has_stopping_criterion = False
        has_stopping_criterion |= self._max_iterations is not None
        has_stopping_criterion |= self._unchanged_max_iterations is not None
        has_stopping_criterion |= self._max_evaluations is not None
        has_stopping_criterion |= self._threshold is not None
        if not has_stopping_criterion:
            raise ValueError("At least one stopping criterion must be set.")

        # Iterations and function evaluations
        iteration = 0
        evaluations = 0

        # Unchanged iterations counter
        unchanged_iterations = 0

        # Choose method to evaluate
        def fun(x):
            return self.call_cost(
                x, cost=self.cost, calculate_grad=self._needs_sensitivities
            )

        # Create evaluator object
        if isinstance(self.cost, BaseJaxCost):
            evaluator = SequentialJaxEvaluator(fun)
        else:
            if self._parallel:
                # Get number of workers
                n_workers = self._n_workers

                # For population based optimisers, don't use more workers than
                # particles!
                if isinstance(self.optimiser, PintsPopulationBasedOptimiser):
                    n_workers = min(n_workers, self.optimiser.population_size())
                evaluator = PintsParallelEvaluator(fun, n_workers=n_workers)
            else:
                evaluator = PintsSequentialEvaluator(fun)

        # Keep track of current best and best-guess scores.
        fb = fg = np.inf

        # Keep track of the last significant change
        f_sig = np.inf

        # Run the ask-and-tell loop
        running = True
        try:
            while running:
                # Ask optimiser for new points
                xs = self.optimiser.ask()

                # Evaluate points
                fs = evaluator.evaluate(xs)

                # Tell optimiser about function values
                self.optimiser.tell(fs)

                # Update the scores
                fb = self.optimiser.f_best()
                fg = self.optimiser.f_guessed()

                # Check for significant changes against the absolute and relative tolerance
                f_new = fg if self._use_f_guessed else fb
                if np.abs(f_new - f_sig) >= np.maximum(
                    self._absolute_tolerance, self._relative_tolerance * np.abs(f_sig)
                ):
                    unchanged_iterations = 0
                    f_sig = f_new
                else:
                    unchanged_iterations += 1

                # Update counts
                evaluations += len(fs)
                iteration += 1
                _fs = [x[0] for x in fs] if self._needs_sensitivities else fs
                self.log_update(
                    iterations=iteration,
                    evaluations=evaluations,
                    x=xs,
                    x_best=self.optimiser.x_best(),
                    cost=_fs,
                    cost_best=fb,
                )

                # Check stopping criteria:
                # Maximum number of iterations
                if (
                    self._max_iterations is not None
                    and iteration >= self._max_iterations
                ):
                    running = False
                    halt_message = (
                        "Maximum number of iterations (" + str(iteration) + ") reached."
                    )

                # Maximum number of iterations without significant change
                halt = (
                    self._unchanged_max_iterations is not None
                    and unchanged_iterations >= self._unchanged_max_iterations
                    and iteration >= self._min_iterations
                )
                if running and halt:
                    running = False
                    halt_message = (
                        "No significant change for "
                        + str(unchanged_iterations)
                        + " iterations."
                    )

                # Maximum number of evaluations
                if (
                    self._max_evaluations is not None
                    and evaluations >= self._max_evaluations
                ):
                    running = False
                    halt_message = (
                        "Maximum number of evaluations ("
                        + str(self._max_evaluations)
                        + ") reached."
                    )

                # Threshold value
                halt = self._threshold is not None and f_new < self._threshold
                if running and halt:
                    running = False
                    halt_message = (
                        "Objective function crossed threshold: "
                        + str(self._threshold)
                        + "."
                    )

                # Error in optimiser
                error = self.optimiser.stop()
                if error:
                    running = False
                    halt_message = str(error)

                elif self._callback is not None:
                    self._callback(iteration - 1, self)

        except (Exception, SystemExit, KeyboardInterrupt):
            # Show last result and exit
            print("\n" + "-" * 40)
            print("Unexpected termination.")
            print("Current score: " + str((fb, fg)))
            print("Current position:")

            # Show current parameters (with any transformation applied)
            for p in self.optimiser.x_guessed():
                print(PintsStrFloat(p))
            print("-" * 40)
            raise

        total_time = time() - start_time

        # Save post-run statistics
        self._evaluations = evaluations
        self._iterations = iteration

        # Get best parameters
        if self._use_f_guessed:
            x = self.optimiser.x_guessed()
            f = self.optimiser.f_guessed()
        else:
            x = self.optimiser.x_best()
            f = self.optimiser.f_best()

        return OptimisationResult(
            optim=self,
            x=x,
            final_cost=f,
            n_iterations=self._iterations,
            n_evaluations=self._evaluations,
            time=total_time,
            message=halt_message,
        )

    def f_guessed_tracking(self):
        """
        Check if f_guessed instead of f_best is being tracked.
        Credit: PINTS

        Returns
        -------
        bool
            True if f_guessed is being tracked, False otherwise.
        """
        return self._use_f_guessed

    def set_f_guessed_tracking(self, use_f_guessed=False):
        """
        Set the method used to track the optimiser progress.
        Credit: PINTS

        Parameters
        ----------
        use_f_guessed : bool, optional
            If True, track f_guessed; otherwise, track f_best (default: False).
        """
        self._use_f_guessed = bool(use_f_guessed)

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
            self._n_workers = PintsParallelEvaluator.cpu_count()
        elif parallel >= 1:
            self._n_workers = int(parallel)
        else:
            self._n_workers = 1

    def set_max_iterations(self, iterations="default"):
        """
        Set the maximum number of iterations as a stopping criterion.
        Credit: PINTS

        Parameters
        ----------
        iterations : int, optional
            The maximum number of iterations to run.
            Set to `None` to remove this stopping criterion.
        """
        if iterations == "default":
            iterations = self.default_max_iterations
        if iterations is not None:
            iterations = int(iterations)
            if iterations < 0:
                raise ValueError("Maximum number of iterations cannot be negative.")
        self._max_iterations = iterations

    def set_min_iterations(self, iterations=2):
        """
        Set the minimum number of iterations as a stopping criterion.

        Parameters
        ----------
        iterations : int, optional
            The minimum number of iterations to run (default: 2).
            Set to `None` to remove this stopping criterion.
        """
        if iterations is not None:
            iterations = int(iterations)
            if iterations < 0:
                raise ValueError("Minimum number of iterations cannot be negative.")
        self._min_iterations = iterations

    def set_max_unchanged_iterations(
        self, iterations=15, absolute_tolerance=1e-5, relative_tolerance=1e-2
    ):
        """
        Set the maximum number of iterations without significant change as a stopping criterion.
        Credit: PINTS

        Parameters
        ----------
        iterations : int, optional
            The maximum number of unchanged iterations to run (default: 15).
            Set to `None` to remove this stopping criterion.
        absolute_tolerance : float, optional
            The minimum significant change (absolute tolerance) in the objective function value
            that resets the unchanged iteration counter (default: 1e-5).
        relative_tolerance : float, optional
            The minimum significant proportional change (relative tolerance) in the objective function
            value that resets the unchanged iteration counter (default: 1e-2).
        """
        if iterations is not None:
            iterations = int(iterations)
            if iterations < 0:
                raise ValueError("Maximum number of iterations cannot be negative.")

        absolute_tolerance = float(absolute_tolerance)
        if absolute_tolerance < 0:
            raise ValueError("Absolute tolerance cannot be negative.")

        relative_tolerance = float(relative_tolerance)
        if relative_tolerance < 0:
            raise ValueError("Relative tolerance cannot be negative.")

        self._unchanged_max_iterations = iterations
        self._absolute_tolerance = absolute_tolerance
        self._relative_tolerance = relative_tolerance

    def set_max_evaluations(self, evaluations=None):
        """
        Set a maximum number of evaluations stopping criterion.
        Credit: PINTS

        Parameters
        ----------
        evaluations : int, optional
            The maximum number of evaluations after which to stop the optimisation
            (default: None).
        """
        if evaluations is not None:
            evaluations = int(evaluations)
            if evaluations < 0:
                raise ValueError("Maximum number of evaluations cannot be negative.")
        self._max_evaluations = evaluations

    def set_threshold(self, threshold=None):
        """
        Adds a stopping criterion, allowing the routine to halt once the
        objective function goes below a set ``threshold``.

        This criterion is disabled by default, but can be enabled by calling
        this method with a valid ``threshold``. To disable it, use
        ``set_threshold(None)``.
        Credit: PINTS

        Parameters
        ----------
        threshold : float, optional
            The threshold below which the objective function value is considered optimal
            (default: None).
        """
        if threshold is None:
            self._threshold = None
        else:
            self._threshold = float(threshold)

    def set_population_size(self, population_size=None):
        """
        Set the population size for population-based optimisers, if specified.
        """
        if isinstance(self.optimiser, PopulationBasedOptimiser):
            self.optimiser.set_population_size(population_size)
