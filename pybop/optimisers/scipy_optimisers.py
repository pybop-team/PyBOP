import numpy as np
from scipy.optimize import differential_evolution, minimize

from .base_optimiser import BaseOptimiser


class SciPyMinimize(BaseOptimiser):
    """
    Adapts SciPy's minimize function for use as an optimization strategy.

    This class provides an interface to various scalar minimization algorithms implemented in SciPy, allowing fine-tuning of the optimization process through method selection and option configuration.

    Parameters
    ----------
    method : str, optional
        The type of solver to use. If not specified, defaults to 'Nelder-Mead'.
        Options: 'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov'.
    bounds : sequence or ``Bounds``, optional
        Bounds for variables as supported by the selected method.
    maxiter : int, optional
        Maximum number of iterations to perform.
    """

    def __init__(self, method=None, bounds=None, maxiter=None, tol=1e-5):
        super().__init__()
        self.method = method
        self.bounds = bounds
        self.num_resamples = 40
        self.tol = tol
        self.options = {}
        self._max_iterations = maxiter

        if self.method is None:
            self.method = "Nelder-Mead"

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

        self.log = [[x0]]
        self.options = {"maxiter": self._max_iterations}

        # Add callback storing history of parameter values
        def callback(x):
            self.log.append([x])

        # Check x0 and resample if required
        self.cost0 = cost_function(x0)
        if np.isinf(self.cost0):
            for i in range(1, self.num_resamples):
                x0 = cost_function.problem.sample_initial_conditions(seed=i)
                self.cost0 = cost_function(x0)
                if not np.isinf(self.cost0):
                    break
            if np.isinf(self.cost0):
                raise Exception("The initial parameter values return an infinite cost.")

        # Scale the cost function and eliminate nan values
        self.inf_count = 0

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

        result = minimize(
            cost_wrapper,
            x0,
            method=self.method,
            bounds=bounds,
            tol=self.tol,
            options=self.options,
            callback=callback,
        )

        return result

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

    def __init__(
        self, bounds=None, strategy="best1bin", maxiter=1000, popsize=15, tol=1e-5
    ):
        super().__init__()
        self.tol = tol
        self.strategy = strategy
        self._max_iterations = maxiter
        self._population_size = popsize

        if bounds is None:
            raise ValueError("Bounds must be specified for differential_evolution.")
        elif not all(
            np.isfinite(value) for sublist in bounds.values() for value in sublist
        ):
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

        self.log = []

        if x0 is not None:
            print(
                "Ignoring x0. Initial conditions are not used for differential_evolution."
            )

        # Add callback storing history of parameter values
        def callback(x, convergence):
            self.log.append([x])

        result = differential_evolution(
            cost_function,
            self.bounds,
            strategy=self.strategy,
            maxiter=self._max_iterations,
            popsize=self._population_size,
            tol=self.tol,
            callback=callback,
        )

        return result

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
