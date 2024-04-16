import numpy as np
from scipy.optimize import differential_evolution, minimize

from .base_optimiser import BaseOptimiser

DEFAULT_SCIPY_MINIMIZE_OPTIONS = dict(
    bounds=None,
    method="Nelder-Mead",
    jac=False,
    tol=1e-5,
    options=dict(),
)
DEFAULT_SCIPY_DIFFERENTIAL_EVOLUTION_OPTIONS = dict(
    bounds=None,
    strategy="best1bin",
    popsize=15,
    tol=1e-5,
    options=dict(),
)


class BaseSciPyOptimiser(BaseOptimiser):
    """
    A base class for defining optimisation methods from the SciPy library.

    Parameters
    ----------
    x0 : array_like
        Initial position from which optimisation will start.
    sigma0 : float, optional
        Initial step size or standard deviation depending on the optimiser.
    bounds : dict, sequence or scipy.optimize.Bounds, optional
        Bounds for variables as supported by the selected method.
    **optimiser_kwargs : optional
        Valid SciPy option keys and their values.
    """

    def __init__(self, x0=None, sigma0=None, bounds=None, **optimiser_kwargs):
        super().__init__(x0, sigma0, bounds)
        self.update_options(**optimiser_kwargs)

    def update_options(self, **optimiser_kwargs):
        """
        Update the optimiser options.

        Parameters
        ----------
        **optimiser_kwargs : optional
            Valid SciPy option keys and their values.
        """
        # Use the first available value for maxiter and remove others
        if "max_iterations" in optimiser_kwargs.keys():
            optimiser_kwargs["maxiter"] = optimiser_kwargs.pop("max_iterations")
        if "options" in optimiser_kwargs.keys():
            if "maxiter" in optimiser_kwargs.keys():
                optimiser_kwargs["options"].pop("maxiter", None)
            elif "maxiter" in optimiser_kwargs["options"].keys():
                optimiser_kwargs["maxiter"] = optimiser_kwargs["options"].pop("maxiter")

        for key, value in optimiser_kwargs.items():
            if key == "bounds":
                self.bounds = value
            elif key == "maxiter":
                self.set_max_iterations(value)
            else:
                self.options[key] = value

    def _run(self, **optimiser_kwargs):
        """
        Internal method to run the optimization using a PyBOP optimiser.

        Parameters
        ----------
        **optimiser_kwargs : optional
            Valid SciPy option keys and their values.

        Returns
        -------
        x : numpy.ndarray
            The best parameter set found by the optimization.
        final_cost : float
            The final cost associated with the best parameters.
        """
        self.update_options(**optimiser_kwargs)

        result = self._run_optimiser()
        self._iterations = result.nit

        return result.x, self._cost_function(result.x)


class SciPyMinimize(BaseSciPyOptimiser):
    """
    Adapts SciPy's minimize function for use as an optimization strategy.

    This class provides an interface to various scalar minimization algorithms implemented in SciPy,
    allowing fine-tuning of the optimization process through method selection and option configuration.

    Parameters
    ----------
    x0 : array_like
        Initial position from which optimisation will start.
    sigma0 : float, optional
        Initial step size or standard deviation depending on the optimiser (default: None).
    bounds : dict, sequence or scipy.optimize.Bounds, optional
        Bounds for variables as supported by the selected method.
    **optimiser_kwargs : optional
        Valid SciPy Minimize option keys and their values. For example, to specify the solver
        use: `method='Nelder-Mead'`. Other options are: 'Nelder-Mead', 'Powell', 'CG', 'BFGS',
        'Newton-CG', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr', 'dogleg', 'trust-ncg',
        'trust-exact', 'trust-krylov'.

    See Also
    --------
    scipy.optimize.minimize : The SciPy method this class is based on.
    """

    def __init__(self, x0=None, sigma0=None, bounds=None, **optimiser_kwargs):
        self.options = DEFAULT_SCIPY_MINIMIZE_OPTIONS
        super().__init__(x0, sigma0, bounds, **optimiser_kwargs)

    def _run_optimiser(self):
        """
        Executes the optimisation process using SciPy's minimize function.

        Parameters
        ----------
        **optimiser_kwargs : optional
            Valid SciPy option keys and their values.

        Returns
        -------
        tuple
            A tuple (x, final_cost) containing the optimized parameters and the value of `cost_function`
            at the optimum.
        """
        self.log = [[self.x0]]

        # Add callback storing history of parameter values
        def callback(x):
            self.log.append([x])

        # Scale the cost function and eliminate nan values
        self._cost0 = self._cost_function(self.x0)
        self.inf_count = 0
        if np.isinf(self._cost0):
            raise Exception("The initial parameter values return an infinite cost.")

        if not self.options["jac"]:

            def cost_wrapper(x):
                cost = self._cost_function(x) / self._cost0
                if np.isinf(cost):
                    self.inf_count += 1
                    cost = 1 + 0.9**self.inf_count  # for fake finite gradient
                return cost
        elif self.options["jac"] is True:

            def cost_wrapper(x):
                return self._cost_function.evaluateS1(x)
        else:
            raise ValueError(
                "Expected the jac option to be either True, False or None."
            )

        # Reformat bounds
        if isinstance(self.bounds, dict):
            bounds = (
                (lower, upper)
                for lower, upper in zip(self.bounds["lower"], self.bounds["upper"])
            )
        else:
            bounds = self.bounds

        # Retrieve maximum iterations
        self.options["options"]["maxiter"] = self._max_iterations

        result = minimize(
            cost_wrapper,
            self.x0,
            method=self.options["method"],
            jac=self.options["jac"],
            bounds=bounds,
            tol=self.options["tol"],
            options=self.options["options"],
            callback=callback,
        )

        return result

    def name(self):
        """
        Provides the name of the optimization strategy.

        Returns
        -------
        str
            The name 'SciPyMinimize'.
        """
        return "SciPyMinimize"


class SciPyDifferentialEvolution(BaseSciPyOptimiser):
    """
    Adapts SciPy's differential_evolution function for global optimization.

    This class provides a global optimization strategy based on differential evolution, useful for
    problems involving continuous parameters and potentially multiple local minima.

    Parameters
    ----------
    bounds : dict, sequence or scipy.optimize.Bounds
        Bounds for variables. Must be provided as it is essential for differential evolution.
    **optimiser_kwargs : optional
        Valid SciPy option keys and their values, such as:
        strategy : str, optional
            The differential evolution strategy to use. Defaults to 'best1bin'.
        maxiter : int, optional
            Maximum number of iterations to perform. Defaults to 1000.
        popsize : int, optional
            The number of individuals in the population. Defaults to 15.

    See Also
    --------
    scipy.optimize.differential_evolution : The SciPy method this class is based on.
    """

    def __init__(self, x0=None, sigma0=None, bounds=None, **optimiser_kwargs):
        self.options = DEFAULT_SCIPY_DIFFERENTIAL_EVOLUTION_OPTIONS
        super().__init__(x0, sigma0, bounds, **optimiser_kwargs)

    def _run_optimiser(self):
        """
        Executes the optimization process using SciPy's differential_evolution function.

        Parameters
        ----------
        **optimiser_kwargs : optional
            Valid SciPy Differential Evolution option keys and their values.

        Returns
        -------
        tuple
            A tuple (x, final_cost) containing the optimized parameters and the value of
            the cost function at the optimum.
        """
        self.log = []

        if self.x0 is not None:
            print(
                "Ignoring x0. Initial conditions are not used for differential_evolution."
            )
        self.x0 = None

        # Add callback storing history of parameter values
        def callback(x, convergence):
            self.log.append([x])

        # Reformat bounds
        if self.bounds is None:
            raise ValueError("Bounds must be specified for differential_evolution.")
        elif isinstance(self.bounds, dict):
            if not all(
                np.isfinite(value)
                for sublist in self.bounds.values()
                for value in sublist
            ):
                raise ValueError("Bounds must be specified for differential_evolution.")
            bounds = [
                (lower, upper)
                for lower, upper in zip(self.bounds["lower"], self.bounds["upper"])
            ]
        else:
            if not all(np.isfinite(value) for pair in self.bounds for value in pair):
                raise ValueError("Bounds must be specified for differential_evolution.")
            bounds = self.bounds

        # Retrieve maximum iterations
        self.options["maxiter"] = self._max_iterations

        result = differential_evolution(
            self._cost_function,
            bounds,
            strategy=self.options["strategy"],
            maxiter=self.options["maxiter"],
            popsize=self.options["popsize"],
            tol=self.options["tol"],
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
            self.options["popsize"] = population_size

    def name(self):
        """
        Provides the name of the optimization strategy.

        Returns
        -------
        str
            The name 'SciPyDifferentialEvolution'.
        """
        return "SciPyDifferentialEvolution"
