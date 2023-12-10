from scipy.optimize import minimize, differential_evolution
from .base_optimiser import BaseOptimiser


class SciPyMinimize(BaseOptimiser):
    """
    Wrapper class for the SciPy optimisation class. Extends the BaseOptimiser class.
    """

    def __init__(self, method=None, bounds=None, maxiter=None):
        super().__init__()
        self.method = method
        self.bounds = bounds
        self.maxiter = maxiter
        if self.maxiter is not None:
            self.options = {"maxiter": self.maxiter}
        else:
            self.options = {}

        if self.method is None:
            self.method = "COBYLA"  # "L-BFGS-B"

    def _runoptimise(self, cost_function, x0, bounds):
        """
        Run the SciPy optimisation method.

        Inputs
        ----------
        cost_function: function for optimising
        method: optimisation algorithm
        x0: initialisation array
        bounds: bounds array
        """

        # Add callback storing history of parameter values
        self.log = [[x0]]

        def callback(x):
            self.log.append([x])

        # Reformat bounds
        if bounds is not None:
            bounds = (
                (lower, upper) for lower, upper in zip(bounds["lower"], bounds["upper"])
            )

        output = minimize(
            cost_function,
            x0,
            method=self.method,
            bounds=bounds,
            options=self.options,
            callback=callback,
        )

        # Get performance statistics
        x = output.x
        final_cost = output.fun

        return x, final_cost

    def needs_sensitivities(self):
        """
        Returns True if the optimiser needs sensitivities.
        """
        return False

    def name(self):
        """
        Returns the name of the optimiser.
        """
        return "SciPyMinimize"


class SciPyDifferentialEvolution(BaseOptimiser):
    """
    Wrapper class for the SciPy differential_evolution optimisation method. Extends the BaseOptimiser class.
    """

    def __init__(self, bounds=None, strategy="best1bin", maxiter=1000, popsize=15):
        super().__init__()
        self.bounds = bounds
        self.strategy = strategy
        self.maxiter = maxiter
        self.popsize = popsize

    def _runoptimise(self, cost_function, x0=None, bounds=None):
        """
        Run the SciPy differential_evolution optimisation method.

        Inputs
        ----------
        cost_function : function
            The objective function to be minimized.
        x0 : array_like
            Initial guess. Only used to determine the dimensionality of the problem.
        bounds : sequence or `Bounds`
            Bounds for variables. There are two ways to specify the bounds:
                1. Instance of `Bounds` class.
                2. Sequence of (min, max) pairs for each element in x, defining the finite lower and upper bounds for the optimizing argument of `cost_function`.
        """

        if bounds is None:
            raise ValueError("Bounds must be specified for differential_evolution.")

        if x0 is not None:
            print(
                "Ignoring x0. Initial conditions are not used for differential_evolution."
            )

        # Add callback storing history of parameter values
        self.log = []

        def callback(x, convergence):
            self.log.append([x])

        # Reformat bounds if necessary
        if isinstance(bounds, dict):
            bounds = [
                (lower, upper) for lower, upper in zip(bounds["lower"], bounds["upper"])
            ]

        output = differential_evolution(
            cost_function,
            bounds,
            strategy=self.strategy,
            maxiter=self.maxiter,
            popsize=self.popsize,
            callback=callback,
        )

        # Get performance statistics
        x = output.x
        final_cost = output.fun

        return x, final_cost

    def needs_sensitivities(self):
        """
        Returns False as differential_evolution does not need sensitivities.
        """
        return False

    def name(self):
        """
        Returns the name of the optimiser.
        """
        return "SciPyDifferentialEvolution"
