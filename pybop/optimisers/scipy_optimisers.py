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
    maxiter=1000,
    tol=1e-5,
    options=dict(),
)


class BaseSciPyOptimiser(BaseOptimiser):
    """
    A base class for defining optimisation methods from the SciPy library.

    Parameters
    ----------
    bounds : dict, sequence or scipy.optimize.Bounds, optional
        Bounds for variables as supported by the selected method.
    **optimiser_kwargs : optional
        Valid SciPy option keys and their values.
    """

    def __init__(self, bounds=None, **optimiser_kwargs):
        super().__init__(bounds)
        self.update_options(**optimiser_kwargs)

    def update_options(self, **optimiser_kwargs):
        """
        Update the optimiser options.
        """
        for key, value in optimiser_kwargs.items():
            if key == "bounds":
                self.bounds = value
            else:
                self.options[key] = value

        # Set optimiser_specific options if required
        self._update_optimiser_options(**optimiser_kwargs)

    def _update_optimiser_options(self, **optimiser_kwargs):
        """
        Update optimiser-specific options. This function should be implemented in
        child classes if required.
        """
        pass
    
    def name(self):
        """
        Provides the name of the optimisation strategy.

        Overwrites the method in Base Optimiser with the method from the PINTS class
        and therefore requires the instance of self to be passed as an input.

        Returns
        -------
        str
            The name given by PINTS.
        """
        return self._pints_class.name(self)


class SciPyMinimize(BaseSciPyOptimiser):
    """
    Adapts SciPy's minimize function for use as an optimization strategy.

    This class provides an interface to various scalar minimization algorithms implemented in SciPy,
    allowing fine-tuning of the optimization process through method selection and option configuration.

    Parameters
    ----------
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

    def __init__(self, bounds=None, **optimiser_kwargs):
        self.options = DEFAULT_SCIPY_MINIMIZE_OPTIONS
        super().__init__(bounds, **optimiser_kwargs)

    def _update_optimiser_options(self, **optimiser_kwargs):
        """
        Update the optimiser-specific options.
        """
        # Overwrite the value of maxiter in the options dictionary
        if "maxiter" in optimiser_kwargs.keys():
            self.options["options"]["maxiter"] = self.options["maxiter"]
            del self.options["maxiter"]

    def _runoptimise(self, cost_function, x0, **optimiser_kwargs):
        """
        Executes the optimisation process using SciPy's minimize function.

        Parameters
        ----------
        cost_function : callable
            The objective function to minimize.
        x0 : array_like
            Initial guess for the parameters.

        Returns
        -------
        tuple
            A tuple (x, final_cost) containing the optimized parameters and the value of `cost_function`
            at the optimum.
        """
        self.update_options(**optimiser_kwargs)
        self.log = [[x0]]

        # Add callback storing history of parameter values
        def callback(x):
            self.log.append([x])

        # Scale the cost function and eliminate nan values
        self.cost0 = cost_function(x0)
        self.inf_count = 0
        if np.isinf(self.cost0):
            raise Exception("The initial parameter values return an infinite cost.")

        if not self.options["jac"]:
            def cost_wrapper(x):
                cost = cost_function(x) / self.cost0
                if np.isinf(cost):
                    self.inf_count += 1
                    cost = 1 + 0.9**self.inf_count  # for fake finite gradient
                return cost
        elif self.options["jac"] is True:
            def cost_wrapper(x):
                return cost_function.evaluateS1(x)
        else:
            raise ValueError("Expected the jac option to be either True, False or None.")

        # Reformat bounds
        if isinstance(self.bounds, dict):
            bounds = (
                (lower, upper)
                for lower, upper in zip(self.bounds["lower"], self.bounds["upper"])
            )
        else:
            bounds = self.bounds

        result = minimize(
            cost_wrapper,
            x0,
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

    def __init__(self, bounds, **optimiser_kwargs):
        self.options = DEFAULT_SCIPY_DIFFERENTIAL_EVOLUTION_OPTIONS
        super().__init__(bounds, **optimiser_kwargs)

    def _update_optimiser_options(self, **optimiser_kwargs):
        """
        Update the optimiser-specific options.
        """
        if self.bounds is None:
            raise ValueError("Bounds must be specified for differential_evolution.")

    def _runoptimise(self, cost_function, x0=None, **optimiser_kwargs):
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
            A tuple (x, final_cost) containing the optimized parameters and the value of
            ``cost_function`` at the optimum.
        """
        self.update_options(**optimiser_kwargs)
        self.log = []

        if x0 is not None:
            print(
                "Ignoring x0. Initial conditions are not used for differential_evolution."
            )

        # Add callback storing history of parameter values
        def callback(x, convergence):
            self.log.append([x])

        # Reformat bounds
        if isinstance(self.bounds, dict):
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

        result = differential_evolution(
            cost_function,
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
