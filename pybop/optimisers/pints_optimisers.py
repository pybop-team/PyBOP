import numpy as np
import pints

from .base_optimiser import BaseOptimiser


class BasePintsOptimiser(BaseOptimiser):
    """
    A base class for defining optimisation methods from the PINTS library.

    Parameters
    ----------
    x0 : array_like
        Initial position from which optimization will start.
    sigma0 : float, optional
        Initial step size or standard deviation depending on the optimiser (default is 0.1).
    bounds : dict, optional
        A dictionary with 'lower' and 'upper' keys containing arrays for lower and upper
        bounds on the parameters.
    **optimiser_kwargs : optional
        Valid PINTS option keys and their values.
    """

    def __init__(
        self, pints_class, x0=None, sigma0=None, bounds=None, **optimiser_kwargs
    ):
        super().__init__(x0, sigma0, bounds)

        # Convert bounds to PINTS boundaries
        if bounds is not None:
            boundaries = pints.RectangularBoundaries(bounds["lower"], bounds["upper"])
        else:
            boundaries = None
        self._boundaries = boundaries

        # Convert x0 to PINTS vector
        self._x0 = pints.vector(self.x0)

        # PyBOP doesn't currently support the PINTS transformation class
        self._transformation = None

        # Create an instance of the PINTS optimiser class
        self.pints_optimiser = pints_class(x0, sigma0, self._boundaries)

        # Check if sensitivities are required
        self._needs_sensitivities = self.pints_optimiser.needs_sensitivities()

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
        # Maximum iterations set in BaseOptimiser

        # Minimum iterations
        self._min_iterations = None
        self.set_min_iterations()

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

        self.update_options(**optimiser_kwargs)

    def update_options(self, **optimiser_kwargs):
        """
        Update the optimiser options.

        Parameters
        ----------
        **optimiser_kwargs : optional
            Valid PINTS option keys and their values.
        """
        for key, value in optimiser_kwargs.items():
            if key == "use_f_guessed":
                self.set_f_guessed_tracking(value)
            elif key == "parallel":
                self.set_parallel(value)
            elif key == "maxiter" or key == "max_iterations":
                self.set_max_iterations(value)
            elif key == "min_iterations":
                self.set_min_iterations(value)
            elif key == "max_unchanged_iterations":
                if "threshold" in optimiser_kwargs.keys():
                    self.set_max_unchanged_iterations(
                        value,
                        optimiser_kwargs["threshold"],
                    )
                else:
                    self.set_max_unchanged_iterations(value)
            elif key == "threshold":
                pass  # only used with unchanged_max_iterations
            elif key == "max_evaluations":
                self.set_max_evaluations(value)
            else:
                raise ValueError(f"Unrecognised or invalid keyword argument: {key}")

    def name(self):
        """
        Provides the name of the optimisation strategy.

        Returns
        -------
        str
            The name given by PINTS.
        """
        return self.pints_optimiser.name()

    def _run(self, **optimiser_kwargs):
        """
        Internal method to run the optimization using a PINTS optimiser.

        Parameters
        ----------
        **optimiser_kwargs : optional
            Valid PINTS option keys and their values.

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
        self.update_options(**optimiser_kwargs)

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
        f = self._cost_function
        if self._needs_sensitivities:
            f = f.evaluateS1

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
        fg_user = (fb, fg) if self._minimising else (-fb, -fg)

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


class GradientDescent(BasePintsOptimiser):
    """
    Implements a simple gradient descent optimization algorithm.

    This class extends the gradient descent optimiser from the PINTS library, designed
    to minimize a scalar function of one or more variables. Note that this optimiser
    does not support boundary constraints.

    Parameters
    ----------
    x0 : array_like
        Initial position from which optimization will start.
    sigma0 : float, optional
        Initial step size (default is 0.1).
    bounds : dict, optional
        Ignored by this optimiser, provided for API consistency.
    **optimiser_kwargs : optional
        Valid PINTS option keys and their values.

    See Also
    --------
    pints.GradientDescent : The PINTS implementation this class is based on.
    """

    def __init__(self, x0=None, sigma0=0.1, bounds=None, **optimiser_kwargs):
        if bounds is not None:
            print("NOTE: Boundaries ignored by Gradient Descent")
        bounds = None  # Bounds ignored in pints.GradientDescent
        super().__init__(pints.GradientDescent, x0, sigma0, bounds, **optimiser_kwargs)

    def set_learning_rate(self, eta):
        """
        Sets the learning rate for this optimiser.
        Credit: PINTS

        Parameters
        ----------
        eta : float
            The learning rate, as a float greater than zero.
        """
        self.pints_optimiser.set_learning_rate(eta)


class Adam(BasePintsOptimiser):
    """
    Implements the Adam optimization algorithm.

    This class extends the Adam optimiser from the PINTS library, which combines
    ideas from RMSProp and Stochastic Gradient Descent with momentum. Note that
    this optimiser does not support boundary constraints.

    Parameters
    ----------
    x0 : array_like
        Initial position from which optimization will start.
    sigma0 : float, optional
        Initial step size (default is 0.1).
    bounds : dict, optional
        Ignored by this optimiser, provided for API consistency.
    **optimiser_kwargs : optional
        Valid PINTS option keys and their values.

    See Also
    --------
    pints.Adam : The PINTS implementation this class is based on.
    """

    def __init__(self, x0=None, sigma0=0.1, bounds=None, **optimiser_kwargs):
        if bounds is not None:
            print("NOTE: Boundaries ignored by Adam")
        bounds = None  # Bounds ignored in pints.Adam
        super().__init__(pints.Adam, x0, sigma0, bounds, **optimiser_kwargs)


class IRPropMin(BasePintsOptimiser):
    """
    Implements the iRpropMin optimization algorithm.

    This class inherits from the PINTS IRPropMin class, which is an optimiser that
    uses resilient backpropagation with weight-backtracking. It is designed to handle
    problems with large plateaus, noisy gradients, and local minima.

    Parameters
    ----------
    x0 : array_like
        Initial position from which optimization will start.
    sigma0 : float, optional
        Initial step size (default is 0.1).
    bounds : dict, optional
        A dictionary with 'lower' and 'upper' keys containing arrays for lower and upper
        bounds on the parameters.
    **optimiser_kwargs : optional
        Valid PINTS option keys and their values.

    See Also
    --------
    pints.IRPropMin : The PINTS implementation this class is based on.
    """

    def __init__(self, x0=None, sigma0=0.1, bounds=None, **optimiser_kwargs):
        super().__init__(pints.IRPropMin, x0, sigma0, bounds, **optimiser_kwargs)


class PSO(BasePintsOptimiser):
    """
    Implements a particle swarm optimization (PSO) algorithm.

    This class extends the PSO optimiser from the PINTS library. PSO is a
    metaheuristic optimization method inspired by the social behavior of birds
    flocking or fish schooling, suitable for global optimization problems.

    Parameters
    ----------
    x0 : array_like
        Initial positions of particles, which the optimization will use.
    sigma0 : float, optional
        Spread of the initial particle positions (default is 0.1).
    bounds : dict, optional
        A dictionary with 'lower' and 'upper' keys containing arrays for lower and upper
        bounds on the parameters.
    **optimiser_kwargs : optional
        Valid PINTS option keys and their values.

    See Also
    --------
    pints.PSO : The PINTS implementation this class is based on.
    """

    def __init__(self, x0=None, sigma0=0.1, bounds=None, **optimiser_kwargs):
        if bounds is not None and not all(
            np.isfinite(value) for sublist in bounds.values() for value in sublist
        ):
            raise ValueError(
                "Either all bounds or no bounds must be set for Pints PSO."
            )
        super().__init__(pints.PSO, x0, sigma0, bounds, **optimiser_kwargs)


class SNES(BasePintsOptimiser):
    """
    Implements the stochastic natural evolution strategy (SNES) optimization algorithm.

    Inheriting from the PINTS SNES class, this optimiser is an evolutionary algorithm
    that evolves a probability distribution on the parameter space, guiding the search
    for the optimum based on the natural gradient of expected fitness.

    Parameters
    ----------
    x0 : array_like
        Initial position from which optimization will start.
    sigma0 : float, optional
        Initial standard deviation of the sampling distribution, defaults to 0.1.
    bounds : dict, optional
        A dictionary with 'lower' and 'upper' keys containing arrays for lower and upper
        bounds on the parameters.
    **optimiser_kwargs : optional
        Valid PINTS option keys and their values.

    See Also
    --------
    pints.SNES : The PINTS implementation this class is based on.
    """

    def __init__(self, x0=None, sigma0=0.1, bounds=None, **optimiser_kwargs):
        super().__init__(pints.SNES, x0, sigma0, bounds, **optimiser_kwargs)


class XNES(BasePintsOptimiser):
    """
    Implements the Exponential Natural Evolution Strategy (XNES) optimiser from PINTS.

    XNES is an evolutionary algorithm that samples from a multivariate normal
    distribution, which is updated iteratively to fit the distribution of successful
    solutions.

    Parameters
    ----------
    x0 : array_like
        The initial parameter vector to optimize.
    sigma0 : float, optional
        Initial standard deviation of the sampling distribution, defaults to 0.1.
    bounds : dict, optional
        A dictionary with 'lower' and 'upper' keys containing arrays for lower and upper
        bounds on the parameters. If ``None``, no bounds are enforced.
    **optimiser_kwargs : optional
        Valid PINTS option keys and their values.

    See Also
    --------
    pints.XNES : PINTS implementation of XNES algorithm.
    """

    def __init__(self, x0=None, sigma0=0.1, bounds=None, **optimiser_kwargs):
        super().__init__(pints.XNES, x0, sigma0, bounds, **optimiser_kwargs)


class NelderMead(BasePintsOptimiser):
    """
    Implements the Nelder-Mead downhill simplex method from PINTS.

    This is a deterministic local optimiser. In most update steps it performs
    either one evaluation, or two sequential evaluations, so that it will not
    typically benefit from parallelisation.

    Parameters
    ----------
    x0 : array_like
        The initial parameter vector to optimize.
    sigma0 : float, optional
        Initial standard deviation of the sampling distribution, defaults to 0.1.
        Does not appear to be used.
    bounds : dict, optional
        Ignored by this optimiser, provided for API consistency.
    **optimiser_kwargs : optional
        Valid PINTS option keys and their values.

    See Also
    --------
    pints.NelderMead : PINTS implementation of Nelder-Mead algorithm.
    """

    def __init__(self, x0=None, sigma0=0.1, bounds=None, **optimiser_kwargs):
        if bounds is not None:
            print("NOTE: Boundaries ignored by NelderMead")
        bounds = None  # Bounds ignored in pints.NelderMead
        super().__init__(pints.NelderMead, x0, sigma0, bounds, **optimiser_kwargs)


class CMAES(BasePintsOptimiser):
    """
    Adapter for the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) optimiser in PINTS.

    CMA-ES is an evolutionary algorithm for difficult non-linear non-convex optimization problems.
    It adapts the covariance matrix of a multivariate normal distribution to capture the shape of
    the cost landscape.

    Parameters
    ----------
    x0 : array_like
        The initial parameter vector to optimize.
    sigma0 : float, optional
        Initial standard deviation of the sampling distribution, defaults to 0.1.
    bounds : dict, optional
        A dictionary with 'lower' and 'upper' keys containing arrays for lower and upper
        bounds on the parameters. If ``None``, no bounds are enforced.
    **optimiser_kwargs : optional
        Valid PINTS option keys and their values.

    See Also
    --------
    pints.CMAES : PINTS implementation of CMA-ES algorithm.
    """

    def __init__(self, x0=None, sigma0=0.1, bounds=None, **optimiser_kwargs):
        if len(x0) == 1:
            raise ValueError(
                "CMAES requires optimisation of >= 2 parameters at once. "
                + "Please choose another optimiser."
            )
        super().__init__(pints.CMAES, x0, sigma0, bounds, **optimiser_kwargs)
