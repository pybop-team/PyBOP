import pybop
import pints
import numpy as np


class Optimisation:
    """
    Optimisation class for PyBOP.
    This class provides functionality for PyBOP optimisers and Pints optimisers.
    args:
        cost: PyBOP cost function
        optimiser: A PyBOP or Pints optimiser
        sigma0: initial step size
        verbose: print optimisation progress

    """

    def __init__(
        self,
        cost,
        optimiser=None,
        sigma0=None,
        verbose=False,
    ):
        self.cost = cost
        self.problem = cost.problem
        self.optimiser = optimiser
        self.verbose = verbose
        self.x0 = cost.problem.x0
        self.bounds = self.problem.bounds
        self.sigma0 = sigma0
        self.log = []

        # Convert x0 to pints vector
        self._x0 = pints.vector(self.x0)

        # PyBOP doesn't currently support the pints transformation class
        self._transformation = None

        # Check if minimising or maximising
        self._minimising = not isinstance(cost, pints.LogPDF)
        if self._minimising:
            self._function = self.cost
        else:
            self._function = pints.ProbabilityBasedError(cost)
        del cost

        # Construct Optimiser
        self.pints = True

        if self.optimiser is None:
            self.optimiser = pybop.CMAES
        elif issubclass(self.optimiser, pints.Optimiser):
            pass
        else:
            self.pints = False

            if issubclass(self.optimiser, pybop.NLoptOptimize):
                self.optimiser = self.optimiser(self.problem.n_parameters)

            elif issubclass(self.optimiser, pybop.SciPyMinimize):
                self.optimiser = self.optimiser()

            else:
                raise ValueError("Unknown optimiser type")

        if self.pints:
            self.optimiser = self.optimiser(self.x0, self.sigma0, self.bounds)

        # Check if sensitivities are required
        self._needs_sensitivities = self.optimiser.needs_sensitivities()

        # Track optimiser's f_best or f_guessed
        self._use_f_guessed = None
        self.set_f_guessed_tracking()

        # Parallelisation
        self._parallel = False
        self._n_workers = 1
        self.set_parallel()

        # User callback
        self._callback = None

        # Define stopping criteria
        # Maximum iterations
        self._max_iterations = None
        self.set_max_iterations()

        # Maximum unchanged iterations
        self._unchanged_threshold = 1  # smallest significant f change
        self._unchanged_max_iterations = None
        self.set_max_unchanged_iterations()

        # Maximum evaluations
        self._max_evaluations = None

        # Threshold value
        self._threshold = None

        # Post-run statistics
        self._evaluations = None
        self._iterations = None

    def run(self):
        """
        Run the optimisation algorithm.
        Selects between PyBOP backend or Pints backend.
        returns:
            x: best parameters
            final_cost: final cost
        """

        if self.pints:
            x, final_cost = self._run_pints()
        elif not self.pints:
            x, final_cost = self._run_pybop()

        return x, final_cost

    def _run_pybop(self):
        """
        Run method for PyBOP based optimisers.
        returns:
            x: best parameters
            final_cost: final cost
        """
        x, final_cost = self.optimiser.optimise(
            cost_function=self.cost,
            x0=self.x0,
            bounds=self.bounds,
        )
        return x, final_cost

    def _run_pints(self):
        """
        Run method for PINTS optimisers.
        This method is heavily based on the run method in the PINTS.OptimisationController class.
        returns:
            x: best parameters
            final_cost: final cost
        """

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
        f = self._function
        if self._needs_sensitivities:
            f = f.evaluateS1

        # Create evaluator object
        if self._parallel:
            # Get number of workers
            n_workers = self._n_workers

            # For population based optimisers, don't use more workers than
            # particles!
            if isinstance(self._optimiser, pints.PopulationBasedOptimiser):
                n_workers = min(n_workers, self._optimiser.population_size())
            evaluator = pints.ParallelEvaluator(f, n_workers=n_workers)
        else:
            evaluator = pints.SequentialEvaluator(f)

        # Keep track of current best and best-guess scores.
        fb = fg = np.inf

        # Internally we always minimise! Keep a 2nd value to show the user.
        fg_user = (fb, fg) if self._minimising else (-fb, -fg)

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
                fg_user = (fb, fg) if self._minimising else (-fb, -fg)

                # Check for significant changes
                f_new = fg if self._use_f_guessed else fb
                if np.abs(f_new - f_sig) >= self._unchanged_threshold:
                    unchanged_iterations = 0
                    f_sig = f_new
                else:
                    unchanged_iterations += 1

                # Update counts
                evaluations += len(fs)
                iteration += 1
                self.log.append(xs)

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
                    self._callback(iteration - 1, self.optimiser)

        except (Exception, SystemExit, KeyboardInterrupt):
            # Show last result and exit
            print("\n" + "-" * 40)
            print("Unexpected termination.")
            print("Current score: " + str(fg_user))
            print("Current position:")

            # Show current parameters
            x_user = self.optimiser.x_guessed()
            if self._transformation is not None:
                x_user = self._transformation.to_model(x_user)
            for p in x_user:
                print(pints.strfloat(p))
            print("-" * 40)
            raise

        if self.verbose:
            print("Halt: " + halt_message)

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

        # Inverse transform search parameters
        if self._transformation is not None:
            x = self._transformation.to_model(x)

        # Return best position and score
        return x, f if self._minimising else -f

    def f_guessed_tracking(self):
        """
        Returns ``True`` if f_guessed instead of f_best is being tracked,
        ``False`` otherwise. See also :meth:`set_f_guessed_tracking`.

        Credit: PINTS
        """
        return self._use_f_guessed

    def set_f_guessed_tracking(self, use_f_guessed=False):
        """
        Sets the method used to track the optimiser progress to
        :meth:`pints.Optimiser.f_guessed()` or
        :meth:`pints.Optimiser.f_best()` (default).

        The tracked ``f`` value is used to evaluate stopping criteria.

        Credit: PINTS
        """
        self._use_f_guessed = bool(use_f_guessed)

    def set_max_evaluations(self, evaluations=None):
        """
        Adds a stopping criterion, allowing the routine to halt after the
        given number of ``evaluations``.

        This criterion is disabled by default. To enable, pass in any positive
        integer. To disable again, use ``set_max_evaluations(None)``.

        Credit: PINTS
        """
        if evaluations is not None:
            evaluations = int(evaluations)
            if evaluations < 0:
                raise ValueError("Maximum number of evaluations cannot be negative.")
        self._max_evaluations = evaluations

    def set_parallel(self, parallel=False):
        """
        Enables/disables parallel evaluation.

        If ``parallel=True``, the method will run using a number of worker
        processes equal to the detected cpu core count. The number of workers
        can be set explicitly by setting ``parallel`` to an integer greater
        than 0.
        Parallelisation can be disabled by setting ``parallel`` to ``0`` or
        ``False``.

        Credit: PINTS
        """
        if parallel is True:
            self._parallel = True
            self._n_workers = pints.ParallelEvaluator.cpu_count()
        elif parallel >= 1:
            self._parallel = True
            self._n_workers = int(parallel)
        else:
            self._parallel = False
            self._n_workers = 1

    def set_max_iterations(self, iterations=10000):
        """
        Adds a stopping criterion, allowing the routine to halt after the
        given number of ``iterations``.

        This criterion is enabled by default. To disable it, use
        ``set_max_iterations(None)``.

        Credit: PINTS
        """
        if iterations is not None:
            iterations = int(iterations)
            if iterations < 0:
                raise ValueError("Maximum number of iterations cannot be negative.")
        self._max_iterations = iterations

    def set_max_unchanged_iterations(self, iterations=200, threshold=1e-11):
        """
        Adds a stopping criterion, allowing the routine to halt if the
        objective function doesn't change by more than ``threshold`` for the
        given number of ``iterations``.

        This criterion is enabled by default. To disable it, use
        ``set_max_unchanged_iterations(None)``.

        Credit: PINTS
        """
        if iterations is not None:
            iterations = int(iterations)
            if iterations < 0:
                raise ValueError("Maximum number of iterations cannot be negative.")

        threshold = float(threshold)
        if threshold < 0:
            raise ValueError("Minimum significant change cannot be negative.")

        self._unchanged_max_iterations = iterations
        self._unchanged_threshold = threshold
