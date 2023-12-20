import pybop
import pints
import numpy as np


class Optimisation:
    """
    A class for conducting optimization using PyBOP or PINTS optimisers.

    Parameters
    ----------
    cost : pybop.BaseCost or pints.ErrorMeasure
        An objective function to be optimized, which can be either a pybop.Cost or PINTS error measure
    optimiser : pybop.Optimiser or subclass of pybop.BaseOptimiser, optional
        An optimiser from either the PINTS or PyBOP framework to perform the optimization (default: None).
    sigma0 : float or sequence, optional
        Initial step size or standard deviation for the optimiser (default: None).
    verbose : bool, optional
        If True, the optimization progress is printed (default: False).

    Attributes
    ----------
    x0 : numpy.ndarray
        Initial parameter values for the optimization.
    bounds : dict
        Dictionary containing the parameter bounds with keys 'lower' and 'upper'.
    n_parameters : int
        Number of parameters in the optimization problem.
    sigma0 : float or sequence
        Initial step size or standard deviation for the optimiser.
    log : list
        Log of the optimization process.
    """

    def __init__(
        self,
        cost,
        optimiser=None,
        sigma0=None,
        verbose=False,
    ):
        self.cost = cost
        self.optimiser = optimiser
        self.verbose = verbose
        self.x0 = cost.x0
        self.bounds = cost.bounds
        self.n_parameters = cost.n_parameters
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
                self.optimiser = self.optimiser(self.n_parameters)

            elif issubclass(
                self.optimiser, (pybop.SciPyMinimize, pybop.SciPyDifferentialEvolution)
            ):
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
        Run the optimization and return the optimized parameters and final cost.

        Returns
        -------
        x : numpy.ndarray
            The best parameter set found by the optimization.
        final_cost : float
            The final cost associated with the best parameters.
        """

        if self.pints:
            x, final_cost = self._run_pints()
        elif not self.pints:
            x, final_cost = self._run_pybop()

        # Store the optimised parameters
        if self.cost.problem is not None:
            self.store_optimised_parameters(x)

        return x, final_cost

    def _run_pybop(self):
        """
        Internal method to run the optimization using a PyBOP optimiser.

        Returns
        -------
        x : numpy.ndarray
            The best parameter set found by the optimization.
        final_cost : float
            The final cost associated with the best parameters.
        """
        x, final_cost = self.optimiser.optimise(
            cost_function=self.cost,
            x0=self.x0,
            bounds=self.bounds,
            maxiter=self._max_iterations,
        )
        self.log = self.optimiser.log

        return x, final_cost

    def _run_pints(self):
        """
        Internal method to run the optimization using a PINTS optimiser.

        Returns
        -------
        x : numpy.ndarray
            The best parameter set found by the optimization.
        final_cost : float
            The final cost associated with the best parameters.

        See Also
        --------
        This method is heavily based on the run method in the PINTS.OptimisationController class.
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

    def set_max_evaluations(self, evaluations=None):
        """
        Set a maximum number of evaluations stopping criterion.
        Credit: PINTS

        Parameters
        ----------
        evaluations : int, optional
            The maximum number of evaluations after which to stop the optimization (default: None).
        """
        if evaluations is not None:
            evaluations = int(evaluations)
            if evaluations < 0:
                raise ValueError("Maximum number of evaluations cannot be negative.")
        self._max_evaluations = evaluations

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
            self._n_workers = pints.ParallelEvaluator.cpu_count()
        elif parallel >= 1:
            self._parallel = True
            self._n_workers = int(parallel)
        else:
            self._parallel = False
            self._n_workers = 1

    def set_max_iterations(self, iterations=1000):
        """
        Set the maximum number of iterations as a stopping criterion.
        Credit: PINTS

        Parameters
        ----------
        iterations : int, optional
            The maximum number of iterations to run (default is 1000).
            Set to `None` to remove this stopping criterion.
        """
        if iterations is not None:
            iterations = int(iterations)
            if iterations < 0:
                raise ValueError("Maximum number of iterations cannot be negative.")
        self._max_iterations = iterations

    def set_max_unchanged_iterations(self, iterations=25, threshold=1e-5):
        """
        Set the maximum number of iterations without significant change as a stopping criterion.
        Credit: PINTS

        Parameters
        ----------
        iterations : int, optional
            The maximum number of unchanged iterations to run (default is 25).
            Set to `None` to remove this stopping criterion.
        threshold : float, optional
            The minimum significant change in the objective function value that resets the unchanged iteration counter (default is 1e-5).
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

    def store_optimised_parameters(self, x):
        """
        Update the problem parameters with optimized values.

        The optimized parameter values are stored within the associated PyBOP parameter class.

        Parameters
        ----------
        x : array-like
            Optimized parameter values.
        """
        for i, param in enumerate(self.cost.problem.parameters):
            param.update(value=x[i])
