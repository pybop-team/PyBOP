import warnings
from time import time
from typing import Union

import numpy as np
from scipy.optimize import Bounds, OptimizeResult, differential_evolution, minimize

from pybop import BaseOptimiser, OptimisationResult, SciPyEvaluator


class BaseSciPyOptimiser(BaseOptimiser):
    """
    A base class for defining optimisation methods from the SciPy library.

    Parameters
    ----------
    x0 : array_like
        Initial position from which optimisation will start.
    bounds : dict, sequence or scipy.optimize.Bounds, optional
        Bounds for variables as supported by the selected method.
    **optimiser_kwargs : optional
        Valid SciPy option keys and their values.
    """

    def __init__(self, cost, **optimiser_kwargs):
        self.num_resamples = 40
        self.key_mapping = {"max_iterations": "maxiter", "population_size": "popsize"}
        super().__init__(cost, **optimiser_kwargs)

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

        # Convert PyBOP keys to SciPy
        for pybop_key, scipy_key in self.key_mapping.items():
            if pybop_key in self.unset_options:
                if scipy_key in self.unset_options:
                    raise Exception(
                        f"The alternative {pybop_key} option was passed in addition to the expected {scipy_key} option."
                    )
                # Rename the key
                self.unset_options[scipy_key] = self.unset_options.pop(pybop_key)

        # Convert bounds to SciPy format
        if isinstance(self.bounds, dict):
            self._scipy_bounds = Bounds(
                self.bounds["lower"], self.bounds["upper"], True
            )
        elif isinstance(self.bounds, Bounds) or self.bounds is None:
            self._scipy_bounds = self.bounds
        else:
            raise TypeError(
                "Bounds provided must be either type dict or SciPy.optimize.bounds object."
            )

    def _run(self):
        """
        Internal method to run the optimisation using a PyBOP optimiser.

        Returns
        -------
        result : pybop.Result
            The result of the optimisation including the optimised parameter values and cost.
        """

        # Choose method to evaluate
        def fun(x):
            return self.call_cost(
                x, cost=self.cost, calculate_grad=self._needs_sensitivities
            )

        # Create evaluator object
        self.evaluator = SciPyEvaluator(fun)

        # Run with timing
        start_time = time()
        result = self._run_optimiser()
        total_time = time() - start_time

        try:
            nit = result.nit
        except AttributeError:
            nit = -1

        return OptimisationResult(
            optim=self,
            x=result.x,
            n_iterations=nit,
            scipy_result=result,
            time=total_time,
            message=result.message,
        )


class SciPyMinimize(BaseSciPyOptimiser):
    """
    Adapts SciPy's minimize function for use as an optimisation strategy.

    This class provides an interface to various scalar minimisation algorithms implemented in SciPy,
    allowing fine-tuning of the optimisation process through method selection and option configuration.

    Parameters
    ----------
    **optimiser_kwargs : optional
        Valid SciPy Minimize option keys and their values:
        x0 : array_like
            Initial position from which optimisation will start.
        method : str
            The optimisation method, options include:
            'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'COBYLA',
            'SLSQP', 'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov'.
        jac : {callable, '2-point', '3-point', 'cs', bool}, optional
            Method for computing the gradient vector.
        hess : {callable, '2-point', '3-point', 'cs', HessianUpdateStrategy}, optional
            Method for computing the Hessian matrix.
        hessp : callable, optional
            Hessian of objective function times an arbitrary vector p.
        bounds : sequence or scipy.optimize.Bounds, optional
            Bounds on variables for L-BFGS-B, TNC, SLSQP, trust-constr methods.
        constraints : {Constraint, dict} or List of {Constraint, dict}, optional
            Constraints definition for constrained optimisation.
        tol : float, optional
            Tolerance for termination.
        options : dict, optional
            Method-specific options. Common options include:
            maxiter : int
                Maximum number of iterations.
            disp : bool
                Set to True to print convergence messages.
            ftol : float
                Function tolerance for termination.
            gtol : float
                Gradient tolerance for termination.
            eps : float
                Step size for finite difference approximation.
            maxfev : int
                Maximum number of function evaluations.
            maxcor : int
                Maximum number of variable metric corrections (L-BFGS-B).

    See Also
    --------
    scipy.optimize.minimize : The SciPy method this class is based on.

    Notes
    -----
    Different optimisation methods may support different options. Consult SciPy's
    documentation for method-specific options and constraints.
    """

    def __init__(self, cost, **optimiser_kwargs):
        optimiser_options = dict(method="Nelder-Mead", jac=False)
        optimiser_options.update(**optimiser_kwargs)
        super().__init__(cost, **optimiser_options)
        self._cost0 = 1.0

    def _set_up_optimiser(self):
        """
        Parse optimiser options.
        """
        # Check and remove any duplicate keywords in self.unset_options
        self._sanitise_inputs()

        # Apply default maxiter
        self._options = dict()
        self._options["options"] = dict()
        self._options["options"]["maxiter"] = self.default_max_iterations

        # Apply additional options and remove them from the options dictionary
        key_list = list(self.unset_options.keys())
        for key in key_list:
            if key in [
                "method",
                "hess",
                "hessp",
                "constraints",
                "tol",
            ]:
                self._options.update({key: self.unset_options.pop(key)})
            elif key == "jac":
                if self.unset_options["jac"] not in [True, False, None]:
                    raise ValueError(
                        f"Expected the jac option to be either True, False or None. Received: {self.unset_options[key]}"
                    )
                self._options.update({key: self.unset_options.pop(key)})
            elif key == "maxiter":
                # Nest this option within an options dictionary for SciPy minimize
                self._options["options"]["maxiter"] = self.unset_options.pop(key)

        if self._options["jac"] is True:
            self._needs_sensitivities = True

    def cost_wrapper(self, x):
        """
        Scale the cost function, preserving the sign convention, and eliminate nan values
        """
        if not self._options["jac"]:
            cost = self.evaluator.evaluate(x)
            self.log_update(x=[x], cost=cost)
            scaled_cost = cost / self._cost0
            if np.isinf(scaled_cost):
                self.inf_count += 1
                scaled_cost = np.sign(cost) * (
                    1 + 0.9**self.inf_count
                )  # for fake finite gradient
            return scaled_cost

        L, dl = self.evaluator.evaluate(x)
        self.log_update(x=[x], cost=L)
        return (L / self._cost0, dl / self._cost0)

    def _run_optimiser(self):
        """
        Executes the optimisation process using SciPy's minimize function.

        Returns
        -------
        result : scipy.optimize.OptimizeResult
            The result of the optimisation including the optimised parameter values and cost.
        """
        self.inf_count = 0

        # Add callback storing history of parameter values
        def base_callback(intermediate_result: Union[OptimizeResult, np.ndarray]):
            """
            Log intermediate optimisation solutions. Depending on the
            optimisation algorithm, intermediate_result may be either
            an OptimizeResult or an array of parameter values, with a
            try/except ensuring both cases are handled correctly.
            """
            if isinstance(intermediate_result, OptimizeResult):
                x_best = intermediate_result.x
                cost_best = intermediate_result.fun * self._cost0
            else:
                x_best = intermediate_result
                result = self.evaluator.evaluate(x_best)
                cost_best = result[0] if self._needs_sensitivities else result

            self.log_update(x_best=x_best, cost_best=cost_best)

        callback = (
            base_callback
            if self._options["method"] != "trust-constr"
            else lambda x, intermediate_result: base_callback(intermediate_result)
        )

        # Compute the absolute initial cost and resample if required
        result = self.evaluator.evaluate(self.x0)
        self._cost0 = np.abs(result[0] if self._needs_sensitivities else result)
        if np.isinf(self._cost0):
            for _i in range(1, self.num_resamples):
                try:
                    self.x0 = self.parameters.rvs(apply_transform=True)
                except AttributeError:
                    warnings.warn(
                        "Parameter does not have a prior distribution. Stopping resampling.",
                        UserWarning,
                        stacklevel=2,
                    )
                    break
                result = self.evaluator.evaluate(self.x0)
                self._cost0 = np.abs(result[0] if self._needs_sensitivities else result)
                if not np.isinf(self._cost0):
                    break
            if np.isinf(self._cost0):
                raise ValueError(
                    "The initial parameter values return an infinite cost."
                )

        return minimize(
            self.cost_wrapper,
            self.x0,
            bounds=self._scipy_bounds,
            callback=callback,
            **self._options,
        )

    def name(self):
        """Provides the name of the optimisation strategy."""
        return "SciPyMinimize"


class SciPyDifferentialEvolution(BaseSciPyOptimiser):
    """
    Adapts SciPy's differential_evolution function for global optimisation.

    This class provides a global optimisation strategy based on differential evolution, useful for
    problems involving continuous parameters and potentially multiple local minima.

    Parameters
    ----------
    bounds : dict, sequence or scipy.optimize.Bounds
        Bounds for variables. Must be provided as it is essential for differential evolution.
        Each element is a tuple (min, max) for the corresponding variable.
    **optimiser_kwargs : optional
        Valid SciPy differential_evolution options:
        strategy : str, optional
            The differential evolution strategy to use. Should be one of:
            - 'best1bin'
            - 'best1exp'
            - 'rand1exp'
            - 'randtobest1exp'
            - 'currenttobest1exp'
            - 'best2exp'
            - 'rand2exp'
            - 'randtobest1bin'
            - 'currenttobest1bin'
            - 'best2bin'
            - 'rand2bin'
            - 'rand1bin'
            Default is 'best1bin'.
        maxiter : int, optional
            Maximum number of generations. Default is 1000.
        popsize : int, optional
            Multiplier for setting the total population size. The population has
            popsize * len(x) individuals. Default is 15.
        tol : float, optional
            Relative tolerance for convergence. Default is 0.01.
        mutation : float or tuple(float, float), optional
            The mutation constant. If specified as a float, should be in [0, 2].
            If specified as a tuple (min, max), dithering is used. Default is (0.5, 1.0).
        recombination : float, optional
            The recombination constant, should be in [0, 1]. Default is 0.7.
        seed : int, optional
            Random seed for reproducibility.
        disp : bool, optional
            Display status messages. Default is False.
        callback : callable, optional
            Called after each iteration with the current result as argument.
        polish : bool, optional
            If True, performs a local optimisation on the solution. Default is True.
        init : str or array-like, optional
            Specify initial population. Can be 'latinhypercube', 'random',
            or an array of shape (M, len(x)).
        atol : float, optional
            Absolute tolerance for convergence. Default is 0.
        updating : {'immediate', 'deferred'}, optional
            If 'immediate', best solution vector is continuously updated within
            a single generation. Default is 'immediate'.
        workers : int or map-like callable, optional
            If workers is an int the population is subdivided into workers
            sections and evaluated in parallel. Default is 1.
        constraints : {NonlinearConstraint, LinearConstraint, Bounds}, optional
            Constraints on the solver.

    See Also
    --------
    scipy.optimize.differential_evolution : The SciPy method this class is based on.

    Notes
    -----
    Differential Evolution is a stochastic population based method that is useful for
    global optimisation problems. At each pass through the population the algorithm mutates
    each candidate solution by mixing with other candidate solutions to create a trial
    candidate. The fitness of all candidates is then evaluated and for each candidate if
    the trial candidate is an improvement, it takes its place in the population for the next
    iteration.
    """

    def __init__(self, cost, **optimiser_kwargs):
        optimiser_options = dict(strategy="best1bin", popsize=15)
        optimiser_options.update(**optimiser_kwargs)
        super().__init__(cost, **optimiser_options)

    def _set_up_optimiser(self):
        """
        Parse optimiser options.
        """
        # Check and remove any duplicate keywords in self.unset_options
        self._sanitise_inputs()

        # Check bounds
        if self._scipy_bounds is None:
            raise ValueError("Bounds must be specified for differential_evolution.")
        else:
            bnds = self._scipy_bounds
            if not (np.isfinite(bnds.lb).all() and np.isfinite(bnds.ub).all()):
                raise ValueError("Bounds must be specified for differential_evolution.")

        # Apply default maxiter and tolerance
        self._options = dict()
        self._options["maxiter"] = self.default_max_iterations
        self._options["tol"] = 1e-5

        # Apply additional options and remove them from the options dictionary
        key_list = list(self.unset_options.keys())
        for key in key_list:
            if key in [
                "strategy",
                "maxiter",
                "popsize",
                "tol",
                "mutation",
                "recombination",
                "seed",
                "disp",
                "polish",
                "init",
                "atol",
                "updating",
                "workers",
                "constraints",
                "tol",
                "integrality",
                "vectorized",
            ]:
                self._options.update({key: self.unset_options.pop(key)})

    def _run_optimiser(self):
        """
        Executes the optimization process using SciPy's differential_evolution function.

        Returns
        -------
        result : scipy.optimize.OptimizeResult
            The result of the optimisation including the optimised parameter values and cost.
        """
        if self.x0 is not None:
            print(
                "Ignoring x0. Initial conditions are not used for differential_evolution."
            )
            self.x0 = None

        # Add callback storing history of parameter values
        def callback(intermediate_result: OptimizeResult):
            self.log_update(
                x_best=intermediate_result.x,
                cost_best=intermediate_result.fun,
            )

        def cost_wrapper(x):
            cost = self.evaluator.evaluate(x)
            self.log_update(x=[x], cost=cost)
            return cost

        return differential_evolution(
            cost_wrapper,
            self._scipy_bounds,
            callback=callback,
            **self._options,
        )

    def name(self):
        """Provides the name of the optimisation strategy."""
        return "SciPyDifferentialEvolution"
