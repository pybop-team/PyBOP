import pybop
import pints
import numpy as np


class Optimisation:
    """
    Optimisation class for PyBOP.
    """

    def __init__(
        self,
        cost,
        optimiser,
        verbose=False,
    ):
        self.cost = cost
        self.problem = cost.problem
        self.optimiser = optimiser
        self.verbose = verbose
        self.x0 = cost.problem.x0
        self.bounds = cost.problem.bounds
        self.fit_parameters = {}
        self.learning_rate = 0.025
        self.max_iterations = 200
        self.max_unchanged_iterations = 10
        self.max_evaluations = None
        self.threshold = None
        self.sigma0 = 0.1

        # Check if minimising or maximising
        self._minimising = not isinstance(cost, pints.LogPDF)

        if self._minimising:
            self._function = cost
        else:
            self._function = pints.ProbabilityBasedError(cost)
        del cost

        # Create optimiser
        if self.optimiser is None or issubclass(self.optimiser, pints.Optimiser):
            self.pints = True
            self.optimiser = self.optimiser or pints.CMAES
            self.optimiser = optimiser(self.x0, self.sigma0, self.bounds)

        elif issubclass(self.optimiser, pybop.NLoptOptimize):
            self.pints = False
            self.optimiser = self.optimiser(self.problem.n_parameters)

        elif issubclass(self.optimiser, pybop.SciPyMinimize):
            self.pints = False
            self.optimiser = self.optimiser()
        else:
            raise ValueError("Optimiser selected is not supported.")

        # Check if sensitivities are required
        self._needs_sensitivities = self.optimiser.needs_sensitivities()

    def run(self):
        """
        Run the optimisation algorithm.
        Selects between PyBOP backend or Pints backend.
        returns:
            x: best parameters
            output: optimiser output
            final_cost: final cost
            num_evals: number of evaluations
        """

        if self.pints:
            x, output, final_cost, num_evals = self._run_pints()
        elif not self.pints:
            x, output, final_cost, num_evals = self._run_pybop()

        return x, output, final_cost, num_evals

    def _run_pybop(self):
        """
        Run method for PyBOP optimisers.
        """
        x, output, final_cost, num_evals = self.optimiser.optimise(
            cost_function=self.cost,
            x0=self.x0,
            bounds=self.bounds,
        )
        return x, output, final_cost, num_evals

    def _run_pints(self):
        """
        Run method for PINTS optimisers.
        This method is based on the run method in the PINTS.OptimisationController class.
        """

        # Check stopping criteria
        has_stopping_criterion = False
        has_stopping_criterion |= self.max_iterations is not None
        has_stopping_criterion |= self.max_unchanged_iterations is not None
        has_stopping_criterion |= self.max_evaluations is not None
        has_stopping_criterion |= self.threshold is not None
        if not has_stopping_criterion:
            raise ValueError("At least one stopping criterion must be set.")

        # Iterations and function evaluations
        iteration = 0
        evaluations = 0

        # Unchanged iterations count (used for stopping or just for
        # information)
        unchanged_iterations = 0

        # Choose method to evaluate
        f = self._function
        if self._needs_sensitivities:
            f = f.evaluateS1

        # Create evaluator object
        evaluator = pints.SequentialEvaluator(f)

        # Keep track of current best and best-guess scores.
        fb = fg = np.inf

        # Internally we always minimise! Keep a 2nd value to show the user.
        fb_user, fg_user = (fb, fg) if self._minimising else (-fb, -fg)

        # Keep track of the last significant change
        f_sig = np.inf

        # Set up progress reporting
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
                fg = self.optimiser.f_guess()
                fb_user, fg_user = (fb, fg) if self._minimising else (-fb, -fg)

                # Check for significant changes
                f_new = fg if self._use_f_guessed else fb
                if np.abs(f_new - f_sig) >= self._unchanged_threshold:
                    unchanged_iterations = 0
                    f_sig = f_new
                else:
                    unchanged_iterations += 1

                # Update evaluation count
                evaluations += len(fs)

                # Update iteration count
                iteration += 1

                #
                # Check stopping criteria
                #

                # Maximum number of iterations
                if (
                    self._max_iterations is not None
                    and iteration >= self._max_iterations
                ):
                    running = False
                    ("Maximum number of iterations (" + str(iteration) + ") reached.")

                # Maximum number of iterations without significant change
                halt = (
                    self._unchanged_max_iterations is not None
                    and unchanged_iterations >= self._unchanged_max_iterations
                )
                if running and halt:
                    running = False
                    (
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
                    (
                        "Maximum number of evaluations ("
                        + str(self._max_evaluations)
                        + ") reached."
                    )

                # Threshold value
                halt = self._threshold is not None and f_new < self._threshold
                if running and halt:
                    running = False
                    (
                        "Objective function crossed threshold: "
                        + str(self._threshold)
                        + "."
                    )

                # Error in optimiser
                error = self.optimiser.stop()
                if error:  # pragma: no cover
                    running = False
                    str(error)

                elif self._callback is not None:
                    self._callback(iteration - 1, self.optimiser)

        except (Exception, SystemExit, KeyboardInterrupt):  # pragma: no cover
            # Unexpected end!
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
