import numpy as np
import pints

from pybop import BaseOptimiser


class BasePintsOptimiser(BaseOptimiser):
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

    def __init__(self, cost, pints_optimiser, **optimiser_kwargs):
        # First set attributes to default values
        self._boundaries = None
        self._needs_sensitivities = None
        self._use_f_guessed = None
        self._parallel = False
        self._n_workers = 1
        self._callback = None
        self._max_iterations = None
        self._min_iterations = 2
        self._unchanged_threshold = 1e-5  # smallest significant f change
        self._unchanged_max_iterations = 15
        self._max_evaluations = None
        self._threshold = None
        self._evaluations = None
        self._iterations = None

        # PyBOP doesn't currently support the PINTS transformation class
        self._transformation = None

        self.pints_optimiser = pints_optimiser
        super().__init__(cost, **optimiser_kwargs)

    def _set_up_optimiser(self):
        """
        Parse optimiser options and create an instance of the PINTS optimiser.
        """
        # Check and remove any duplicate keywords in self.unset_options
        self._sanitise_inputs()

        # Create an instance of the PINTS optimiser class
        if issubclass(self.pints_optimiser, pints.Optimiser):
            self.pints_optimiser = self.pints_optimiser(
                self.x0, sigma0=self.sigma0, boundaries=self._boundaries
            )
        else:
            raise ValueError(
                "The pints_optimiser is not a recognised PINTS optimiser class."
            )

        # Check if sensitivities are required
        self._needs_sensitivities = self.pints_optimiser.needs_sensitivities()

        # Apply default maxiter
        self.set_max_iterations()

        # Apply additional options and remove them from options
        key_list = list(self.unset_options.keys())
        for key in key_list:
            if key == "use_f_guessed":
                self.set_f_guessed_tracking(self.unset_options.pop(key))
            elif key == "parallel":
                self.set_parallel(self.unset_options.pop(key))
            elif key == "max_iterations":
                self.set_max_iterations(self.unset_options.pop(key))
            elif key == "min_iterations":
                self.set_min_iterations(self.unset_options.pop(key))
            elif key == "max_unchanged_iterations":
                if "threshold" in self.unset_options.keys():
                    self.set_max_unchanged_iterations(
                        self.unset_options.pop(key),
                        self.unset_options.pop("threshold"),
                    )
                else:
                    self.set_max_unchanged_iterations(self.unset_options.pop(key))
            elif key == "threshold":
                pass  # only used with unchanged_max_iterations
            elif key == "max_evaluations":
                self.set_max_evaluations(self.unset_options.pop(key))

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

        # Check for duplicate keywords
        expected_keys = [
            "max_iterations",
            "popsize",
            "threshold",
        ]
        alternative_keys = ["maxiter", "population_size", "tol"]
        for exp_key, alt_key in zip(expected_keys, alternative_keys):
            if alt_key in self.unset_options.keys():
                if exp_key in self.unset_options.keys():
                    raise Exception(
                        "The alternative {alt_key} option was passed in addition to the expected {exp_key} option."
                    )
                else:  # rename
                    self.unset_options[exp_key] = self.unset_options.pop(alt_key)

        # Convert bounds to PINTS boundaries
        if self.bounds is not None:
            if issubclass(
                self.pints_optimiser,
                (pints.GradientDescent, pints.Adam, pints.NelderMead),
            ):
                print(f"NOTE: Boundaries ignored by {self.pints_optimiser}")
                self.bounds = None
            elif issubclass(self.pints_optimiser, pints.PSO):
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

    def name(self):
        """
        Provides the name of the optimisation strategy.

        Returns
        -------
        str
            The name given by PINTS.
        """
        return self.pints_optimiser.name()

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

        # Unchanged iterations counter
        unchanged_iterations = 0

        # Choose method to evaluate
        if self._needs_sensitivities:

            def f(x):
                L, dl = self.cost.evaluateS1(x)
                return (L, dl) if self.minimising else (-L, -dl)
        else:

            def f(x, grad=None):
                return self.cost(x, grad) if self.minimising else -self.cost(x, grad)

        # Create evaluator object
        if self._parallel:
            # Get number of workers
            n_workers = self._n_workers

            # For population based optimisers, don't use more workers than
            # particles!
            if isinstance(self.pints_optimiser, pints.PopulationBasedOptimiser):
                n_workers = min(n_workers, self.pints_optimiser.population_size())
            evaluator = pints.ParallelEvaluator(f, n_workers=n_workers)
        else:
            evaluator = pints.SequentialEvaluator(f)

        # Keep track of current best and best-guess scores.
        fb = fg = np.inf

        # Internally we always minimise! Keep a 2nd value to show the user.
        fg_user = (fb, fg) if self.minimising else (-fb, -fg)

        # Keep track of the last significant change
        f_sig = np.inf

        # Run the ask-and-tell loop
        running = True
        try:
            while running:
                # Ask optimiser for new points
                xs = self.pints_optimiser.ask()

                # Evaluate points
                fs = evaluator.evaluate(xs)

                # Tell optimiser about function values
                self.pints_optimiser.tell(fs)

                # Update the scores
                fb = self.pints_optimiser.f_best()
                fg = self.pints_optimiser.f_guessed()
                fg_user = (fb, fg) if self.minimising else (-fb, -fg)

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
                error = self.pints_optimiser.stop()
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
            x_user = self.pints_optimiser.x_guessed()
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
            x = self.pints_optimiser.x_guessed()
            f = self.pints_optimiser.f_guessed()
        else:
            x = self.pints_optimiser.x_best()
            f = self.pints_optimiser.f_best()

        # Inverse transform search parameters
        if self._transformation is not None:
            x = self._transformation.to_model(x)

        # Store result
        final_cost = f if self.minimising else -f
        self.result = Result(x=x, final_cost=final_cost, nit=self._iterations)

        # Return best position and its cost
        return x, final_cost

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

    def __init__(self, x=None, final_cost=None, nit=None):
        self.x = x
        self.final_cost = final_cost
        self.nit = nit
