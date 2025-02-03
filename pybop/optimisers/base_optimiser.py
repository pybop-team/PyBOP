import warnings
from copy import deepcopy
from typing import Optional, Union

import jax.numpy as jnp
import numpy as np
from pybamm import Solution

from pybop import BaseCost, BaseLikelihood, Inputs, Parameter, Parameters


class BaseOptimiser:
    """
    A base class for defining optimisation methods.

    This class serves as a base class for creating optimisers. It provides a basic structure for
    an optimisation algorithm, including the initial setup and a method stub for performing the
    optimisation process. Child classes should override _set_up_optimiser and the _run method with
    a specific algorithm.

    Parameters
    ----------
    cost : pybop.BaseCost or pints.ErrorMeasure
        An objective function to be optimised, which can be either a pybop.Cost or PINTS error measure
    **optimiser_kwargs : optional
            Valid option keys and their values.

    Attributes
    ----------
    x0 : numpy.ndarray
        Initial parameter values for the optimisation.
    bounds : dict
        Dictionary containing the parameter bounds with keys 'lower' and 'upper'.
    sigma0 : float or sequence
        Initial step size or standard deviation around ``x0``. Either a scalar value (one
        standard deviation for all coordinates) or an array with one entry per dimension.
        Not all methods will use this information.
    verbose : bool, optional
        If True, the optimisation progress is printed (default: False).
    physical_viability : bool, optional
        If True, the feasibility of the optimised parameters is checked (default: False).
    allow_infeasible_solutions : bool, optional
        If True, infeasible parameter values will be allowed in the optimisation (default: True).
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
        self.x0 = optimiser_kwargs.get("x0", [])
        self.log = dict(x=[], x_best=[], x_search=[], x0=[], cost=[], cost_best=[])
        self.bounds = None
        self.sigma0 = 0.02
        self.verbose = True
        self._transformation = None
        self._needs_sensitivities = False
        self._minimising = True
        self.physical_viability = False
        self.allow_infeasible_solutions = False
        self.default_max_iterations = 1000
        self.result = None

        if isinstance(cost, BaseCost):
            self.cost = cost
            self.parameters = deepcopy(self.cost.parameters)
            self._transformation = self.cost.transformation
            self.set_allow_infeasible_solutions()
            self._minimising = self.cost.minimising

        else:
            try:
                cost_test = cost(self.x0)
                warnings.warn(
                    "The cost is not an instance of pybop.BaseCost, but let's continue "
                    "assuming that it is a callable function to be minimised.",
                    UserWarning,
                    stacklevel=2,
                )
                self.cost = cost
                for i, value in enumerate(self.x0):
                    self.parameters.add(
                        Parameter(name=f"Parameter {i}", initial_value=value)
                    )

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
        # Set initial values, if x0 is None, initial values are unmodified.
        self.parameters.update(initial_values=self.unset_options.pop("x0", None))
        self.log_update(x0=self.parameters.reset_initial_value())
        self.x0 = self.parameters.reset_initial_value(apply_transform=True)

        # Set default bounds (for all or no parameters)
        bounds = self.unset_options.pop(
            "bounds", self.parameters.get_bounds(apply_transform=True)
        )
        if isinstance(bounds, (np.ndarray, list)):
            self.parameters.update(bounds=bounds)
            bounds = self.parameters.get_bounds(apply_transform=True)
        self.bounds = bounds

        # Set default initial standard deviation (for all or no parameters)
        self.sigma0 = self.unset_options.pop(
            "sigma0", self.parameters.get_sigma0(apply_transform=True) or self.sigma0
        )

        # Set other options
        self.verbose = self.unset_options.pop("verbose", self.verbose)
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

    def cost_call(
        self,
        x: Union[Inputs, list],
        calculate_grad: bool = False,
    ) -> Union[float, tuple[float, np.ndarray]]:
        """
        Call the cost function to minimise, applying any given transformation to the
        input parameters.

        Parameters
        ----------
        x : Inputs or list-like
            The input parameters for which the cost and optionally the gradient
            will be computed.
        calculate_grad : bool, optional, default=False
            If True, both the cost and gradient will be computed. Otherwise, only the
            cost is computed.

        Returns
        -------
        float or tuple
            - If `calculate_grad` is False, returns the computed cost (float).
            - If `calculate_grad` is True, returns a tuple containing the cost (float)
              and the gradient (np.ndarray).
        """
        return self.cost(
            x,
            calculate_grad=calculate_grad,
            apply_transform=True,
            for_optimiser=True,
        )

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
                self.x0 = self.parameters.rvs(1, apply_transform=True)
                self.parameters.update(initial_values=self.x0)
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

    def log_update(self, x=None, x_best=None, cost=None, cost_best=None, x0=None):
        """
        Update the log with new values.

        Parameters
        ----------
        x : list or array-like, optional
            Parameter values (default: None).
        x_best : list or array-like, optional
            Parameter values corresponding to the best cost yet (default: None).
        cost : list, optional
            Cost values corresponding to x (default: None).
        cost_best
            Cost values corresponding to x_best (default: None).
        """

        def convert_to_list(array_like):
            """Helper function to convert input to a list, if necessary."""
            if isinstance(array_like, (list, tuple, np.ndarray, jnp.ndarray)):
                return list(array_like)
            elif isinstance(array_like, (int, float)):
                return [array_like]
            else:
                raise TypeError("Input must be a list, tuple, or numpy array")

        def apply_transformation(values):
            """Apply transformation if it exists."""
            if self._transformation:
                return [self._transformation.to_model(value) for value in values]
            return values

        if x is not None:
            x = convert_to_list(x)
            self.log["x_search"].extend(x)
            x = apply_transformation(x)
            self.log["x"].extend(x)

        if x_best is not None:
            x_best = apply_transformation([x_best])
            self.log["x_best"].extend(x_best)

        if cost is not None:
            cost = convert_to_list(cost)
            cost = [
                internal_cost * (1 if self.minimising else -1) for internal_cost in cost
            ]
            self.log["cost"].extend(cost)

        if cost_best is not None:
            cost_best = convert_to_list(cost_best)
            cost_best = [
                internal_cost * (1 if self.minimising else -1)
                for internal_cost in cost_best
            ]
            self.log["cost_best"].extend(cost_best)

        if x0 is not None:
            self.log["x0"].extend(x0)

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

    @property
    def transformation(self):
        return self._transformation

    @property
    def minimising(self):
        return self._minimising


class OptimisationResult:
    """
    Stores the result of the optimisation.

    Attributes
    ----------
    x : ndarray
        The solution of the optimisation.
    final_cost : float
        The cost associated with the solution x.
    n_iterations : int
        Number of iterations performed by the optimiser.
    scipy_result : scipy.optimize.OptimizeResult, optional
        The result obtained from a SciPy optimiser.
    pybamm_solution: pybamm.Solution or list[pybamm.Solution], optional
        The best solution object(s) obtained from the optimisation.
    """

    def __init__(
        self,
        optim: BaseOptimiser,
        x: Union[Inputs, np.ndarray] = None,
        final_cost: Optional[float] = None,
        sensitivities: Optional[dict] = None,
        n_iterations: Optional[int] = None,
        n_evaluations: Optional[int] = None,
        time: Optional[float] = None,
        scipy_result=None,
    ):
        self.optim = optim
        self.cost = self.optim.cost
        self.minimising = self.optim.minimising
        self._transformation = self.optim.transformation
        self.n_runs = 0
        self._best_run = None
        self._x = []
        self._final_cost = []
        self._sensitivities = None
        self._fisher = []
        self._n_iterations = []
        self._n_evaluations = []
        self._scipy_result = []
        self._time = []
        self._x0 = []
        self._pybamm_solution = []

        if x is not None:
            # Transform the parameter values and update the sign of any final cost
            # coming directly from an optimiser
            x = self._transformation.to_model(x) if self._transformation else x
            final_cost = (
                final_cost * (1 if self.minimising else -1)
                if final_cost is not None
                else self.cost(x)
            )
            x0 = (
                self.optim.parameters.initial_value()
                if isinstance(self.optim, BaseOptimiser)
                else None
            )

            # Evaluate the problem once more to update the solution
            try:
                self.cost(x)
                pybamm_solution = self.cost.pybamm_solution
            except Exception:
                warnings.warn(
                    "Failed to evaluate the model with best fit parameters.",
                    UserWarning,
                    stacklevel=2,
                )
                pybamm_solution = None

            # Calculate Fisher Information if Likelihood
            if isinstance(self.cost, BaseLikelihood):
                fisher = self.cost.observed_fisher(x)
                diag_fish = np.diag(fisher) if fisher is not None else None
            else:
                diag_fish = None

            self._extend(
                x=[x],
                final_cost=[final_cost],
                fisher=[diag_fish],
                n_iterations=[n_iterations],
                n_evaluations=[n_evaluations],
                time=[time],
                scipy_result=[scipy_result],
                x0=[x0],
                pybamm_solution=[pybamm_solution],
            )

    def add_result(self, result):
        """Add a preprocessed OptimisationResult."""
        self._extend(
            x=result._x,  # noqa: SLF001
            final_cost=result._final_cost,  # noqa: SLF001
            fisher=result._fisher,  # noqa: SLF001
            n_iterations=result._n_iterations,  # noqa: SLF001
            n_evaluations=result._n_evaluations,  # noqa: SLF001
            time=result._time,  # noqa: SLF001
            scipy_result=result._scipy_result,  # noqa: SLF001
            x0=result._x0,  # noqa: SLF001
            pybamm_solution=result._pybamm_solution,  # noqa: SLF001
        )

    def _extend(
        self,
        x: Union[list[Inputs], list[np.ndarray]],
        final_cost: list[float],
        fisher: list,
        n_iterations: list[int],
        n_evaluations: list[int],
        time: list[float],
        scipy_result: list,
        x0: list,
        pybamm_solution: list[Solution],
    ):
        self.n_runs += len(final_cost)
        self._x.extend(x)
        self._final_cost.extend(final_cost)
        self._fisher.extend(fisher)
        self._n_iterations.extend(n_iterations)
        self._n_evaluations.extend(n_evaluations)
        self._scipy_result.extend(scipy_result)
        self._time.extend(time)
        self._x0.extend(x0)
        self._pybamm_solution.extend(pybamm_solution)

        # Check that there is a finite cost and update best run
        self.check_for_finite_cost()
        self._best_run = self._final_cost.index(
            min(self._final_cost) if self.minimising else max(self._final_cost)
        )

        # Check that the best parameters are physically viable
        self.check_physical_viability(self.x_best)

    def check_for_finite_cost(self) -> None:
        """
        Validate the optimised parameters and ensure they produce a finite cost value.

        Raises:
            ValueError: If the optimised parameters do not produce a finite cost value.
        """
        if not any(np.isfinite(self._final_cost)):
            raise ValueError("Optimised parameters do not produce a finite cost value")

    def check_physical_viability(self, x):
        """
        Check if the optimised parameters are physically viable.

        Parameters
        ----------
        x : array-like
            Optimised parameter values.
        """
        if (
            not isinstance(self.cost, BaseCost)
            or self.cost.problem is None
            or self.cost.problem.model is None
        ):
            warnings.warn(
                "No model within problem class, can't check physical viability.",
                UserWarning,
                stacklevel=2,
            )
            return

        if self.cost.problem.model.check_params(
            inputs=x, allow_infeasible_solutions=False
        ):
            return
        else:
            warnings.warn(
                "Optimised parameters are not physically viable! \nConsider retrying the optimisation"
                " with a non-gradient-based optimiser and the option allow_infeasible_solutions=False",
                UserWarning,
                stacklevel=2,
            )

    def __str__(self) -> str:
        """
        A string representation of the OptimisationResult object.

        Returns:
            str: A formatted string containing optimisation result information.
        """
        # Format the sensitivities
        self.sense_format = ""
        if self._sensitivities:
            self.sense_format = ""
            for value, conf in zip(
                self._sensitivities["ST"], self._sensitivities["ST_conf"]
            ):
                self.sense_format += f" {value:.3f} ± {conf:.3f},"

        return (
            f"OptimisationResult:\n"
            f"  Best result from {self.n_runs} run(s).\n"
            f"  Initial parameters: {self.x0_best}\n"
            f"  Optimised parameters: {self.x_best}\n"
            f"  Total-order sensitivities:{self.sense_format}\n"
            f"  Diagonal Fisher Information entries: {self.fisher_best}\n"
            f"  Final cost: {self.final_cost_best}\n"
            f"  Optimisation time: {self.time_best} seconds\n"
            f"  Number of iterations: {self.n_iterations_best}\n"
            f"  Number of evaluations: {self.n_evaluations_best}\n"
            f"  SciPy result available: {'Yes' if self.scipy_result_best else 'No'}\n"
            f"  PyBaMM Solution available: {'Yes' if self.pybamm_solution else 'No'}"
        )

    def average_iterations(self) -> Optional[float]:
        """Calculates the average number of iterations across all runs."""
        return np.mean(self._n_iterations)

    def total_runtime(self) -> Optional[float]:
        """Calculates the total runtime across all runs."""
        return np.sum(self._time)

    def _get_single_or_all(self, attr):
        value = getattr(self, attr)
        return value[0] if len(value) == 1 else value

    @property
    def x(self):
        return self._get_single_or_all("_x")

    @property
    def x_best(self):
        return self._x[self._best_run] if self._best_run is not None else None

    @property
    def x0(self):
        return self._get_single_or_all("_x0")

    @property
    def x0_best(self):
        return self._x0[self._best_run] if self._best_run is not None else None

    @property
    def final_cost(self):
        return self._get_single_or_all("_final_cost")

    @property
    def final_cost_best(self):
        return self._final_cost[self._best_run] if self._best_run is not None else None

    @property
    def fisher(self):
        return self._get_single_or_all("_fisher")

    @property
    def sensitivities(self):
        return self._get_single_or_all("_sensitivities")

    @sensitivities.setter
    def sensitivities(self, obj: dict):
        self._sensitivities = obj

    @property
    def fisher_best(self):
        return self._fisher[self._best_run] if self._best_run is not None else None

    @property
    def n_iterations(self):
        return self._get_single_or_all("_n_iterations")

    @property
    def n_iterations_best(self):
        return (
            self._n_iterations[self._best_run] if self._best_run is not None else None
        )

    @property
    def n_evaluations(self):
        return self._get_single_or_all("_n_evaluations")

    @property
    def n_evaluations_best(self):
        return (
            self._n_evaluations[self._best_run] if self._best_run is not None else None
        )

    @property
    def scipy_result(self):
        return self._get_single_or_all("_scipy_result")

    @property
    def scipy_result_best(self):
        return (
            self._scipy_result[self._best_run] if self._best_run is not None else None
        )

    @property
    def pybamm_solution(self):
        return self._get_single_or_all("_pybamm_solution")

    @property
    def pybamm_solution_best(self):
        return (
            self._pybamm_solution[self._best_run]
            if self._best_run is not None
            else None
        )

    @property
    def time(self):
        return self._get_single_or_all("_time")

    @property
    def time_best(self):
        return self._time[self._best_run] if self._best_run is not None else None
