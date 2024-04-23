import numpy as np
import pints

from pybop import Optimisation

DEFAULT_PINTS_OPTIMISER_OPTIONS = dict(
    _boundaries=None,
    _x0=None,
    _transformation=None,  # PyBOP doesn't currently support the PINTS transformation class
    _use_f_guessed=None,
    _parallel=False,
    _n_workers=1,
    _callback=None,
    _min_iterations=2,
    _unchanged_threshold=1e-5,  # smallest significant f change
    _unchanged_max_iterations=15,
    _max_evaluations=None,
    _threshold=None,
    _evaluations=None,
    _iterations=None,
)


class BasePintsOptimiser(Optimisation):
    """
    A base class for defining optimisation methods from the PINTS library.

    Parameters
    ----------
    **optimiser_kwargs : optional
        Valid PINTS option keys and their values, for example:
        x0 : array_like
            Initial position from which optimization will start.
        sigma0 : float
            Initial step size or standard deviation depending on the optimiser.
        bounds : dict
            A dictionary with 'lower' and 'upper' keys containing arrays for lower and
            upper bounds on the parameters.
    """

    def __init__(self, cost, pints_method, **optimiser_kwargs):
        self.__dict__.update(DEFAULT_PINTS_OPTIMISER_OPTIONS)
        self.pints_method = pints_method
        super().__init__(cost, **optimiser_kwargs)

        # Create  an instance of the PINTS optimiser class
        self.initialise_method()

    def _set_options(self, **optimiser_kwargs):
        """
        Update the optimiser options and remove the corresponding entries from the
        optimiser_kwargs dictionary in advance of passing to the parent class.

        Parameters
        ----------
        **optimiser_kwargs : optional
            Valid PINTS option keys and their values.

        Returns
        -------
        optimiser_kwargs : dict
            Remaining option keys and their values.
        """
        reinit_required = False

        key_list = list(optimiser_kwargs.keys())
        for key in key_list:
            if key in ["x0", "sigma0", "bounds"]:
                self.__dict__.update({key: optimiser_kwargs.pop(key)})
                reinit_required = True
            elif key == "use_f_guessed":
                self.set_f_guessed_tracking(optimiser_kwargs.pop(key))
            elif key == "parallel":
                self.set_parallel(optimiser_kwargs.pop(key))
            elif key == "maxiter" or key == "max_iterations":
                self.set_max_iterations(optimiser_kwargs.pop(key))
            elif key == "min_iterations":
                self.set_min_iterations(optimiser_kwargs.pop(key))
            elif key == "max_unchanged_iterations":
                if "threshold" in optimiser_kwargs.keys():
                    self.set_max_unchanged_iterations(
                        optimiser_kwargs.pop(key),
                        optimiser_kwargs.pop("threshold"),
                    )
                else:
                    self.set_max_unchanged_iterations(optimiser_kwargs.pop(key))
            elif key == "threshold":
                pass  # only used with unchanged_max_iterations
            elif key == "max_evaluations":
                self.set_max_evaluations(optimiser_kwargs.pop(key))

        if reinit_required:
            self.initialise_method()

        return optimiser_kwargs

    def initialise_method(self):
        """
        Creates an instance of the PINTS optimiser class.
        """
        if issubclass(self.pints_method, pints.Optimiser):
            self.method = self.pints_method(self.x0, self.sigma0, self._boundaries)
        else:
            raise ValueError("The pints_method is not recognised as a PINTS optimiser.")

        # Convert x0 to PINTS vector
        self._x0 = pints.vector(self.x0)

        # Convert bounds to PINTS boundaries
        if self.bounds is not None:
            if issubclass(
                self.pints_method, (pints.GradientDescent, pints.Adam, pints.NelderMead)
            ):
                print(f"NOTE: Boundaries ignored by {self.pints_method}")
                self.bounds = None
                self._boundaries = None
            elif issubclass(self.pints_method, pints.PSO):
                if not all(
                    np.isfinite(value)
                    for sublist in self.bounds.values()
                    for value in sublist
                ):
                    raise ValueError(
                        "Either all bounds or no bounds must be set for Pints PSO."
                    )
            else:
                self._boundaries = pints.RectangularBoundaries(
                    self.bounds["lower"], self.bounds["upper"]
                )
        else:
            self._boundaries = None

        # Check if sensitivities are required
        self._needs_sensitivities = self.method.needs_sensitivities()

    def name(self):
        """
        Provides the name of the optimisation strategy.

        Returns
        -------
        str
            The name given by PINTS.
        """
        return self.method.name()

    def _run(self):
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

        # Empty result
        self.result = Result()

        # Unchanged iterations counter
        unchanged_iterations = 0

        # Choose method to evaluate
        f = self.cost
        if self._needs_sensitivities:
            f = f.evaluateS1

        # Create evaluator object
        if self._parallel:
            # Get number of workers
            n_workers = self._n_workers

            # For population based optimisers, don't use more workers than
            # particles!
            if isinstance(self.method, pints.PopulationBasedOptimiser):
                n_workers = min(n_workers, self.method.population_size())
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
                xs = self.method.ask()

                # Evaluate points
                fs = evaluator.evaluate(xs)

                # Tell optimiser about function values
                self.method.tell(fs)

                # Update the scores
                fb = self.method.f_best()
                fg = self.method.f_guessed()
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
                error = self.method.stop()
                if error:
                    running = False
                    halt_message = str(error)

                elif self._callback is not None:
                    self._callback(iteration - 1, self)

        except (Exception, SystemExit, KeyboardInterrupt):
            # Show last result and exit
            print("\n" + "-" * 40)
            print("Unexpected termination.")
            print("Current score: " + str(fg_user))
            print("Current position:")

            # Show current parameters
            x_user = self.method.x_guessed()
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
            x = self.method.x_guessed()
            f = self.method.f_guessed()
        else:
            x = self.method.x_best()
            f = self.method.f_best()

        # Inverse transform search parameters
        if self._transformation is not None:
            x = self._transformation.to_model(x)

        # Store result
        self.result.x = x
        self.result.final_cost = f
        self.result.nit = self._iterations

        # Return best position and the score used internally,
        # i.e the negative log-likelihood in the case of
        # self._minimising = False
        return x, f

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
        if parallel is True:
            self._parallel = True
            self._n_workers = pints.ParallelEvaluator.cpu_count()
        elif parallel >= 1:
            self._parallel = True
            self._n_workers = int(parallel)
        else:
            self._parallel = False
            self._n_workers = 1

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

    def set_max_unchanged_iterations(self, iterations=15, threshold=1e-5):
        """
        Set the maximum number of iterations without significant change as a stopping criterion.
        Credit: PINTS

        Parameters
        ----------
        iterations : int, optional
            The maximum number of unchanged iterations to run (default: 15).
            Set to `None` to remove this stopping criterion.
        threshold : float, optional
            The minimum significant change in the objective function value that resets the
            unchanged iteration counter (default: 1e-5).
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


class Result:
    """
    Stores the result of the optimisation.

    Attributes
    ----------
    x : ndarray
        The solution of the optimisation.
    final_cost : float
        The cost associated with the solution x.
    nit : int
        Number of iterations performed by the optimiser.

    """

    def __init__(self):
        self.x = None
        self.final_cost = None
        self.nit = None
