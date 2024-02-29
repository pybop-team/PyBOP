from scipy.optimize import minimize, differential_evolution
from .base_optimiser import BaseOptimiser
import numpy as np


class SciPyMinimize(BaseOptimiser):
    """
    Adapts SciPy's minimize function for use as an optimization strategy.

    This class provides an interface to various scalar minimization algorithms implemented in SciPy, allowing fine-tuning of the optimization process through method selection and option configuration.

    Parameters
    ----------
    method : str, optional
        The type of solver to use. If not specified, defaults to 'COBYLA'.
    bounds : sequence or ``Bounds``, optional
        Bounds for variables as supported by the selected method.
    maxiter : int, optional
        Maximum number of iterations to perform.
    """

    def __init__(self, method=None, bounds=None, maxiter=None):
        super().__init__()
        self.method = method
        self.bounds = bounds
        self.options = {}
        self._max_iterations = maxiter

        if self.method is None:
            self.method = "COBYLA"  # "L-BFGS-B"

    def _runoptimise(self, cost_function, x0):
        """
        Executes the optimization process using SciPy's minimize function.

        Parameters
        ----------
        cost_function : callable
            The objective function to minimize.
        x0 : array_like
            Initial guess for the parameters.

        Returns
        -------
        tuple
            A tuple (x, final_cost) containing the optimized parameters and the value of `cost_function` at the optimum.
        """

        # Add callback storing history of parameter values
        self.log = [[x0]]

        def callback(x):
            self.log.append([x])

        # Scale the cost function and eliminate nan values
        self.cost0 = cost_function(x0)
        self.inf_count = 0
        if np.isinf(self.cost0):
            raise Exception("The initial parameter values return an infinite cost.")

        def cost_wrapper(x):
            cost = cost_function(x) / self.cost0
            if np.isinf(cost):
                self.inf_count += 1
                cost = 1 + 0.9**self.inf_count  # for fake finite gradient
            return cost

        # Reformat bounds
        if self.bounds is not None:
            bounds = (
                (lower, upper)
                for lower, upper in zip(self.bounds["lower"], self.bounds["upper"])
            )

        # Set max iterations
        if self._max_iterations is not None:
            self.options = {"maxiter": self._max_iterations}
        else:
            self.options.pop("maxiter", None)

        output = minimize(
            cost_wrapper,
            x0,
            method=self.method,
            bounds=bounds,
            options=self.options,
            callback=callback,
        )

        # Get performance statistics
        x = output.x
        final_cost = cost_function(x)

        return x, final_cost

    def needs_sensitivities(self):
        """
        Determines if the optimization algorithm requires gradient information.

        Returns
        -------
        bool
            False, indicating that gradient information is not required.
        """
        return False

    def name(self):
        """
        Provides the name of the optimization strategy.

        Returns
        -------
        str
            The name 'SciPyMinimize'.
        """
        return "SciPyMinimize"


class SciPyDifferentialEvolution(BaseOptimiser):
    """
    Adapts SciPy's differential_evolution function for global optimization.

    This class provides a global optimization strategy based on differential evolution, useful for problems involving continuous parameters and potentially multiple local minima.

    Parameters
    ----------
    bounds : sequence or ``Bounds``
        Bounds for variables. Must be provided as it is essential for differential evolution.
    strategy : str, optional
        The differential evolution strategy to use. Defaults to 'best1bin'.
    maxiter : int, optional
        Maximum number of iterations to perform. Defaults to 1000.
    popsize : int, optional
        The number of individuals in the population. Defaults to 15.
    """

    def __init__(self, bounds=None, strategy="best1bin", maxiter=1000, popsize=15):
        super().__init__()
        self.strategy = strategy
        self._max_iterations = maxiter
        self._population_size = popsize

        if bounds is None:
            raise ValueError("Bounds must be specified for differential_evolution.")
        elif not all(np.isfinite(value) for sublist in bounds.values() for value in sublist):
            raise ValueError("Bounds must be specified for differential_evolution.")
        elif isinstance(bounds, dict):
            bounds = [
                (lower, upper) for lower, upper in zip(bounds["lower"], bounds["upper"])
            ]
        self.bounds = bounds

    def _runoptimise(self, cost_function, x0=None):
        """
        Executes the optimization process using SciPy's differential_evolution function.

        Parameters
        ----------
        cost_function : callable
            The objective function to minimize.
        x0 : array_like, optional
            Ignored parameter, provided for API consistency.

        Returns
        -------
        tuple
            A tuple (x, final_cost) containing the optimized parameters and the value of ``cost_function`` at the optimum.
        """

        if x0 is not None:
            print(
                "Ignoring x0. Initial conditions are not used for differential_evolution."
            )

        # Add callback storing history of parameter values
        self.log = []

        def callback(x, convergence):
            self.log.append([x])

        output = differential_evolution(
            cost_function,
            self.bounds,
            strategy=self.strategy,
            maxiter=self._max_iterations,
            popsize=self._population_size,
            callback=callback,
        )

        # Get performance statistics
        x = output.x
        final_cost = output.fun

        return x, final_cost

    def set_population_size(self, population_size=None):
        """
        Sets a population size to use in this optimisation.
        Credit: PINTS

        """
        # Check population size or set using heuristic
        if population_size is not None:
            population_size = int(population_size)
            if population_size < 1:
                raise ValueError("Population size must be at least 1.")
            self._population_size = population_size

    def needs_sensitivities(self):
        """
        Determines if the optimization algorithm requires gradient information.

        Returns
        -------
        bool
            False, indicating that gradient information is not required for differential evolution.
        """
        return False

    def name(self):
        """
        Provides the name of the optimization strategy.

        Returns
        -------
        str
            The name 'SciPyDifferentialEvolution'.
        """
        return "SciPyDifferentialEvolution"
