import warnings

import numpy as np
from scipy.optimize import differential_evolution, minimize

from pybop import Optimisation

DEFAULT_SCIPY_MINIMIZE_OPTIONS = dict(
    method="Nelder-Mead",
    jac=False,
    tol=1e-5,
    options=dict(),
)
DEFAULT_SCIPY_DIFFERENTIAL_EVOLUTION_OPTIONS = dict(
    strategy="best1bin",
    popsize=15,
    tol=1e-5,
    options=dict(),
)


class BaseSciPyOptimiser(Optimisation):
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

    def __init__(self, cost, **optimiser_kwargs):
        super().__init__(cost, **optimiser_kwargs)

    def _set_options(self, **optimiser_kwargs):
        """
        Update the optimiser options and remove the corresponding entries from the
        optimiser_kwargs dictionary in advance of passing to the parent class.

        Parameters
        ----------
        **optimiser_kwargs : optional
            Valid SciPy option keys and their values.

        Returns
        -------
        optimiser_kwargs : dict
            Remaining option keys and their values.
        """
        # Unpack nested values from SciPy options dictionary
        if "options" in optimiser_kwargs.keys():
            options_list = list(optimiser_kwargs["options"].keys())
            for key in options_list:
                if key not in optimiser_kwargs.keys():
                    optimiser_kwargs[key] = optimiser_kwargs["options"].pop(key)
                else:
                    optimiser_kwargs["options"].pop(key)  # remove entry

        # Keep max_iterations in preference to maxiter
        if "max_iterations" in optimiser_kwargs.keys():
            optimiser_kwargs["maxiter"] = optimiser_kwargs.pop("max_iterations")

        key_list = list(optimiser_kwargs.keys())
        for key in key_list:
            if key == "maxiter":
                self.set_max_iterations(optimiser_kwargs.pop(key))
            elif key in ["method", "jac", "tol", "options", "strategy", "popsize"]:
                self.__dict__.update({key: optimiser_kwargs.pop(key)})

        return optimiser_kwargs

    def _run(self):
        """
        Internal method to run the optimization using a PyBOP optimiser.

        Returns
        -------
        x : numpy.ndarray
            The best parameter set found by the optimization.
        final_cost : float
            The final cost associated with the best parameters.
        """
        self.result = self._run_optimiser()

        self.result.final_cost = self.cost(self.result.x)
        self._iterations = self.result.nit

        return self.result.x, self.result.final_cost

    def set_max_unchanged_iterations(self, *args, **kwargs):
        """
        Raise a warning that this stopping criterion is not used by this optimiser.
        """
        invalid_criteria_warning = "The maximum unchanged iterations stopping criteria is not used by the SciPy optimisers."
        warnings.warn(invalid_criteria_warning, UserWarning)


class SciPyMinimize(BaseSciPyOptimiser):
    """
    Adapts SciPy's minimize function for use as an optimization strategy.

    This class provides an interface to various scalar minimization algorithms implemented in SciPy,
    allowing fine-tuning of the optimization process through method selection and option configuration.

    Parameters
    ----------
    **optimiser_kwargs : optional
        Valid SciPy Minimize option keys and their values, For example:
        x0 : array_like
            Initial position from which optimisation will start.
        bounds : dict, sequence or scipy.optimize.Bounds
            Bounds for variables as supported by the selected method.
        method : str
            The optimisation method, options include:
            'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'COBYLA',
            'SLSQP', 'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov'.

    See Also
    --------
    scipy.optimize.minimize : The SciPy method this class is based on.
    """

    def __init__(self, cost, **optimiser_kwargs):
        self.__dict__.update(DEFAULT_SCIPY_MINIMIZE_OPTIONS)
        super().__init__(cost, **optimiser_kwargs)

    def _run_optimiser(self):
        """
        Executes the optimisation process using SciPy's minimize function.

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
        self._cost0 = self.cost(self.x0)
        self.inf_count = 0
        if np.isinf(self._cost0):
            raise Exception("The initial parameter values return an infinite cost.")

        if not self.jac:

            def cost_wrapper(x):
                cost = self.cost(x) / self._cost0
                if np.isinf(cost):
                    self.inf_count += 1
                    cost = 1 + 0.9**self.inf_count  # for fake finite gradient
                return cost
        elif self.jac is True:

            def cost_wrapper(x):
                return self.cost.evaluateS1(x)
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
        self.options["maxiter"] = self._max_iterations

        result = minimize(
            cost_wrapper,
            self.x0,
            method=self.method,
            jac=self.jac,
            bounds=bounds,
            tol=self.tol,
            options=self.options,
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
        Valid SciPy option keys and their values, for example:
        strategy : str
            The differential evolution strategy to use.
        maxiter : int
            Maximum number of iterations to perform.
        popsize : int
            The number of individuals in the population.

    See Also
    --------
    scipy.optimize.differential_evolution : The SciPy method this class is based on.
    """

    def __init__(self, cost, **optimiser_kwargs):
        self.__dict__.update(DEFAULT_SCIPY_DIFFERENTIAL_EVOLUTION_OPTIONS)
        super().__init__(cost, **optimiser_kwargs)

    def _run_optimiser(self):
        """
        Executes the optimization process using SciPy's differential_evolution function.

        Returns
        -------
        tuple
            A tuple (x, final_cost) containing the optimized parameters and the value of
            the cost function at the optimum.
        """
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

        result = differential_evolution(
            self.cost,
            bounds,
            strategy=self.strategy,
            maxiter=self._max_iterations,
            popsize=self.popsize,
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
            self.popsize = population_size

    def name(self):
        """
        Provides the name of the optimization strategy.

        Returns
        -------
        str
            The name 'SciPyDifferentialEvolution'.
        """
        return "SciPyDifferentialEvolution"
