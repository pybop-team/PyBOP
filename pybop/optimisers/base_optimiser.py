import warnings
from copy import deepcopy
from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

from pybop import (
    BaseCost,
    CostInterface,
    OptimisationResult,
    Parameter,
    Parameters,
)


@dataclass
class OptimisationLog:
    """Stores optimisation progress data."""

    iterations: list[int] = field(default_factory=list)
    evaluations: list[int] = field(default_factory=list)
    x: list[list[float]] = field(default_factory=list)
    x_best: list[list[float]] = field(default_factory=list)
    x_search: list[list[float]] = field(default_factory=list)
    x0: list[list[float]] = field(default_factory=list)
    cost: list[float] = field(default_factory=list)
    cost_best: list[float] = field(default_factory=list)


class BaseOptimiser(CostInterface):
    """
    A base class for defining optimisation methods. Optimisers perform minimisation of the cost
    function; maximisation may be performed instead using the option invert_cost=True.

    This class serves as a base class for creating optimisers. It provides a basic structure for
    an optimisation algorithm, including the initial setup and a method stub for performing the
    optimisation process. Child classes should override _set_up_optimiser and the _run method with
    a specific algorithm.

    Parameters
    ----------
    cost : pybop.BaseCost or callable
        An objective function to be optimised, which can be either a pybop.Cost or callable function.
    **optimiser_kwargs : optional
        Valid option keys and their values, for example:
        x0 : numpy.ndarray
            Initial values of the parameters for the optimisation.
        bounds : dict
            Dictionary containing bounds for the parameters with keys 'lower' and 'upper'.
        sigma0 : float or sequence
            Initial step size or standard deviation in the (search) parameters. Either a scalar value
            (same for all coordinates) or an array with one entry per dimension.
            Not all methods will use this information.
        verbose : bool, optional
            If True, the optimisation progress and final result is printed (default: False).
        verbose_print_rate : int, optional
            The frequency in iterations to print the optimisation progress (default: 50).
        physical_viability : bool, optional
            If True, the feasibility of the optimised parameters is checked (default: False).
        allow_infeasible_solutions : bool, optional
            If True, infeasible parameter values will be allowed in the optimisation (default: True).

    Attributes
    ----------
    log : dict
        A log of the parameter values tried during the optimisation and associated costs.
    """

    def __init__(
        self,
        cost,
        **optimiser_kwargs,
    ):
        # First set attributes to default values
        self.parameters = Parameters()
        self.x0 = optimiser_kwargs.get("x0", None)
        self.bounds = None
        self.sigma0 = 0.02
        self.verbose = False
        self.verbose_print_rate = 50
        self._needs_sensitivities = False
        self.physical_viability = False
        self.allow_infeasible_solutions = False
        self.default_max_iterations = 1000
        self.result = None
        self._iter_count = 0
        self.log = OptimisationLog()
        transformation = None
        invert_cost = False

        if isinstance(cost, BaseCost):
            self.cost = cost
            self.parameters = deepcopy(self.cost.parameters)
            transformation = self.parameters.construct_transformation()
            self.set_allow_infeasible_solutions()
            invert_cost = not self.cost.minimising

        else:
            try:
                x0 = optimiser_kwargs.get("x0", [])
                for i, value in enumerate(x0):
                    self.parameters.add(
                        Parameter(name=f"Parameter {i}", initial_value=value)
                    )
                cost_test = cost(x0)
                warnings.warn(
                    "The cost is not an instance of pybop.BaseCost, but let's continue "
                    "assuming that it is a callable function to be minimised.",
                    UserWarning,
                    stacklevel=2,
                )
                self.cost = cost

            except Exception as e:
                raise Exception(
                    "The cost is not a recognised cost object or function."
                ) from e

            if not np.isscalar(cost_test) or not np.isreal(cost_test):
                raise TypeError(
                    f"Cost returned {type(cost_test)}, not a scalar numeric value."
                )

        if len(self.parameters) == 0:
            raise ValueError("There are no parameters to optimise.")

        super().__init__(transformation=transformation, invert_cost=invert_cost)

        self.unset_options = optimiser_kwargs
        self.unset_options_store = optimiser_kwargs.copy()
        self.set_base_options()
        self._set_up_optimiser()

        # Throw a warning if any options remain
        if self.unset_options:
            warnings.warn(
                f"Unrecognised keyword arguments: {self.unset_options} will not be used.",
                UserWarning,
                stacklevel=2,
            )

    def set_base_options(self):
        """
        Update the base optimiser options and remove them from the options dictionary.
        """
        # Set initial search-space parameter values
        x0 = self.unset_options.pop("x0", None)
        if x0 is not None:
            self.parameters.update(initial_values=x0)
        self.x0 = self.parameters.reset_initial_value(apply_transform=True)

        # Set the search-space parameter bounds (for all or no parameters)
        bounds = self.unset_options.pop("bounds", self.parameters.get_bounds())
        if bounds is not None:
            self.parameters.update(bounds=bounds)
            bounds = self.parameters.get_bounds(apply_transform=True)
        self.bounds = bounds  # can be None or current parameter bounds

        # Set default initial standard deviation (for all or no parameters)
        self.sigma0 = self.unset_options.pop(
            "sigma0", self.parameters.get_sigma0(apply_transform=True) or self.sigma0
        )

        # Set other options
        self.verbose = self.unset_options.pop("verbose", self.verbose)
        self.verbose_print_rate = self.unset_options.pop(
            "verbose_print_rate", self.verbose_print_rate
        )
        if "allow_infeasible_solutions" in self.unset_options.keys():
            self.set_allow_infeasible_solutions(
                self.unset_options.pop("allow_infeasible_solutions")
            )

        # Set multistart
        self.multistart = self.unset_options.pop("multistart", 1)

        # Parameter sensitivities
        self.compute_sensitivities = self.unset_options.pop(
            "compute_sensitivities", False
        )
        self.n_samples_sensitivity = self.unset_options.pop(
            "n_sensitivity_samples", 256
        )

    def _set_up_optimiser(self):
        """
        Parse optimiser options and prepare the optimiser.

        This method should be implemented by child classes.

        Raises
        ------
        NotImplementedError
            If the method has not been implemented by the subclass.
        """
        raise NotImplementedError

    def run(self):
        """
        Run the optimisation and return the optimised parameters and final cost.

        Returns
        -------
        results: OptimisationResult
            The pybop optimisation result class.
        """
        self.result = OptimisationResult(optim=self)

        for i in range(self.multistart):
            if i >= 1:
                self.unset_options = self.unset_options_store.copy()
                self.parameters.update(initial_values=self.parameters.rvs(1))
                self.x0 = self.parameters.reset_initial_value(apply_transform=True)
                self._set_up_optimiser()

            self.result.add_result(self._run())

        # Store the optimised parameters
        self.parameters.update(values=self.result.x_best)

        # Compute sensitivities
        self.result.sensitivities = self._parameter_sensitivities()

        if self.verbose:
            print(self.result)

        return self.result

    def _run(self):
        """
        Contains the logic for the optimisation algorithm.

        This method should be implemented by child classes to perform the actual optimisation.

        Raises
        ------
        NotImplementedError
            If the method has not been implemented by the subclass.
        """
        raise NotImplementedError

    def _parameter_sensitivities(self):
        if not self.compute_sensitivities:
            return None

        return self.cost.sensitivity_analysis(self.n_samples_sensitivity)

    def log_update(
        self,
        iterations=None,
        evaluations=None,
        x=None,
        x_best=None,
        cost=None,
        cost_best=None,
        x0=None,
    ):
        """
        Update the log with new values.

        Parameters
        ----------
        iterations : list or array-like, optional
            Iteration indices to log (default: None).
        evaluations: list or array-like, optional
            Evaluation indices to log (default: None).
        x : list or array-like, optional
            Parameter values (default: None).
        x_best : list or array-like, optional
            Parameter values corresponding to the best cost yet (default: None).
        cost : list, optional
            Cost values corresponding to x (default: None).
        cost_best : list, optional
            Cost values corresponding to x_best (default: None).
        x0 : list or array-like, optional
            Initial parameter values (default: None).
        """
        # Update logs for each provided parameter
        self._update_log_entry("iterations", iterations)
        self._update_log_entry("evaluations", evaluations)

        if x is not None:
            x_list = self._to_list(x)
            self.log.x_search.extend(x_list)
            transformed_x = self.transform_list_of_values(x_list)
            self.log.x.extend(transformed_x)

        if x_best is not None:
            transformed_x_best = self.transform_values(x_best)
            self.log.x_best.append(transformed_x_best)

        if cost is not None:
            self.log.cost.extend(self._inverts_cost(self._to_list(cost)))

        if cost_best is not None:
            self.log.cost_best.extend(self._inverts_cost(self._to_list(cost_best)))

        # Verbose output
        self._print_verbose_output()
        self._iter_count += 1

    def _update_log_entry(self, key, value):
        """Update a log entry if the value is provided."""
        if value is not None:
            getattr(self.log, key).extend(self._to_list(value))

    def _to_list(self, array_like):
        """Convert input to a list."""
        if isinstance(array_like, (list, tuple, np.ndarray, jnp.ndarray)):
            return list(array_like)
        return [array_like]

    def _print_verbose_output(self):
        """Print verbose optimization information if enabled."""
        if not self.verbose:
            return

        latest_iter = (
            self.log.iterations[-1] if self.log.iterations else self._iter_count
        )

        # Only print on first 10 iterations, then every Nth iteration
        if latest_iter > 10 and latest_iter % self.verbose_print_rate != 0:
            return

        latest_eval = self.log.evaluations[-1] if self.log.evaluations else "N/A"
        latest_x_best = self.log.x_best[-1] if self.log.x_best else "N/A"
        latest_cost_best = self.log.cost_best[-1] if self.log.cost_best else "N/A"

        print(
            f"Iter: {latest_iter} | Evals: {latest_eval} | "
            f"Best Values: {latest_x_best} | Best Cost: {latest_cost_best} |"
        )

    def name(self):
        """
        Returns the name of the optimiser, to be overwritten by child classes.

        Returns
        -------
        str
            The name of the optimiser
        """
        raise NotImplementedError  # pragma: no cover

    def set_allow_infeasible_solutions(self, allow: bool = True):
        """
        Set whether to allow infeasible solutions or not.

        Parameters
        ----------
        iterations : bool, optional
            Whether to allow infeasible solutions.
        """
        # Set whether to allow infeasible locations
        self.physical_viability = allow
        self.allow_infeasible_solutions = allow

        if (
            isinstance(self.cost, BaseCost)
            and self.cost.problem is not None
            and self.cost.problem.model is not None
        ):
            self.cost.problem.model.allow_infeasible_solutions = (
                self.allow_infeasible_solutions
            )
        else:
            # Turn off this feature as there is no model
            self.physical_viability = False
            self.allow_infeasible_solutions = False

    @property
    def needs_sensitivities(self):
        return self._needs_sensitivities
