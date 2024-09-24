import warnings
from typing import Union

import numpy as np
from scipy.optimize import Bounds, OptimizeResult, differential_evolution, minimize

from pybop import BaseOptimiser, Result


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
        super().__init__(cost, **optimiser_kwargs)
        self.num_resamples = 40

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
        expected_keys = ["maxiter", "popsize"]
        alternative_keys = ["max_iterations", "population_size"]
        for exp_key, alt_key in zip(expected_keys, alternative_keys):
            if alt_key in self.unset_options.keys():
                if exp_key in self.unset_options.keys():
                    raise Exception(
                        "The alternative {alt_key} option was passed in addition to the expected {exp_key} option."
                    )
                else:  # rename
                    self.unset_options[exp_key] = self.unset_options.pop(alt_key)

        # Convert bounds to SciPy format
        if isinstance(self.bounds, dict):
            self._scipy_bounds = Bounds(
                self.bounds["lower"], self.bounds["upper"], True
            )
        elif isinstance(self.bounds, list):
            lb, ub = zip(*self.bounds)
            self._scipy_bounds = Bounds(lb, ub, True)
        elif isinstance(self.bounds, Bounds) or self.bounds is None:
            self._scipy_bounds = self.bounds
        else:
            raise TypeError(
                "Bounds provided must be either type dict, list or SciPy.optimize.bounds object."
            )

    def _run(self):
        """
        Internal method to run the optimization using a PyBOP optimiser.

        Returns
        -------
        result : pybop.Result
            The result of the optimisation including the optimised parameter values and cost.
        """
        result = self._run_optimiser()

        try:
            nit = result.nit
        except AttributeError:
            nit = -1

        return Result(
            x=self._transformation.to_model(result.x)
            if self._transformation
            else result.x,
            final_cost=self.cost(result.x, apply_transform=True),
            n_iterations=nit,
            scipy_result=result,
        )


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

    def cost_wrapper(self, x):
        """
        Scale the cost function, preserving the sign convention, and eliminate nan values
        """
        self.log_update(x=[x])

        if not self._options["jac"]:
            cost = self.cost(x, apply_transform=True) / self._cost0
            if np.isinf(cost):
                self.inf_count += 1
                cost = 1 + 0.9**self.inf_count  # for fake finite gradient
            return cost if self.minimising else -cost

        L, dl = self.cost(x, calculate_grad=True, apply_transform=True)
        return (L, dl) if self.minimising else (-L, -dl)

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
            a OptimizeResult or an array of parameter values, with a
            try/except ensuring both cases are handled correctly.
            """
            if isinstance(intermediate_result, OptimizeResult):
                x_best = intermediate_result.x
                cost = intermediate_result.fun
            else:
                x_best = intermediate_result
                cost = self.cost(x_best, apply_transform=True)

            self.log_update(
                x_best=x_best,
                cost=(-1 if not self.minimising else 1) * cost * self._cost0,
            )

        callback = (
            base_callback
            if self._options["method"] != "trust-constr"
            else lambda x, intermediate_result: base_callback(intermediate_result)
        )

        # Compute the absolute initial cost and resample if required
        self._cost0 = np.abs(self.cost(self.x0, apply_transform=True))
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
                self._cost0 = np.abs(self.cost(self.x0, apply_transform=True))
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
                cost=intermediate_result.fun
                if self.minimising
                else -intermediate_result.fun,
            )

        def cost_wrapper(x):
            self.log_update(x=[x])
            return (
                self.cost(x, apply_transform=True)
                if self.minimising
                else -self.cost(x, apply_transform=True)
            )

        return differential_evolution(
            cost_wrapper,
            self._scipy_bounds,
            callback=callback,
            **self._options,
        )

    def name(self):
        """
        Provides the name of the optimization strategy.

        Returns
        -------
        str
            The name 'SciPyDifferentialEvolution'.
        """
        return "SciPyDifferentialEvolution"
