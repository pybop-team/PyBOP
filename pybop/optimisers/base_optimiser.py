import warnings
from typing import Optional, Union

import numpy as np

from pybop import BaseCost, Inputs, Parameter, Parameters


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
            self.parameters = self.cost.parameters
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
        self.bounds = self.unset_options.pop(
            "bounds", self.parameters.get_bounds(apply_transform=True)
        )

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
        self.parameters.update(values=self.result.x)

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
            if isinstance(array_like, (list, tuple, np.ndarray)):
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
            hasattr(self.cost, "problem")
            and hasattr(self.cost.problem, "model")
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
    """

    def __init__(
        self,
        optim: BaseOptimiser,
        x: Union[Inputs, np.ndarray] = None,
        final_cost: Optional[float] = None,
        n_iterations: Optional[int] = None,
        time: Optional[float] = None,
        scipy_result=None,
    ):
        self.optim = optim
        self.cost = self.optim.cost
        self.minimising = self.optim.minimising
        self._transformation = self.optim._transformation  # noqa: SLF001

        self.n_runs = 0
        self._best_run = None
        self._x = []
        self._final_cost = []
        self._n_iterations = []
        self._scipy_result = []
        self._time = []
        self._x0 = []

        if x is not None:
            # Transform the parameter values and update the sign of any final cost
            # coming directly from an optimiser
            x = self._transformation.to_model(x) if self._transformation else x
            final_cost = (
                final_cost * (1 if self.minimising else -1)
                if final_cost is not None
                else self.cost(x)
            )
            self._append(
                x=x,
                final_cost=final_cost,
                n_iterations=n_iterations,
                time=time,
                scipy_result=scipy_result,
            )

    def add_result(self, result=None):
        """Add a preprocessed OptimisationResult."""
        for i, x in enumerate(result._x):
            self._append(
                x=x,
                final_cost=result._final_cost[i],
                n_iterations=result._n_iterations[i],
                time=result._time[i],
                scipy_result=result._scipy_result[i],
            )

    def _append(
        self,
        x: Union[Inputs, np.ndarray] = None,
        final_cost: Optional[float] = None,
        n_iterations: Optional[int] = None,
        time: Optional[float] = None,
        scipy_result=None,
    ):
        self.n_runs += 1
        self._x.append(x)
        self._final_cost.append(final_cost)
        self._n_iterations.append(n_iterations)
        self._scipy_result.append(scipy_result)
        self._time.append(time)
        if isinstance(self.optim, BaseOptimiser):
            self._x0.append(self.optim.parameters.initial_value())

        # Check that the parameters produce finite cost, and are physically viable
        self._validate_parameters()
        self.check_physical_viability(x)

        # Update best run
        self._best_run = self._final_cost.index(
            min(self._final_cost) if self.minimising else max(self._final_cost)
        )

    def _validate_parameters(self) -> None:
        """
        Validate the optimised parameters and ensure they produce a finite cost value.

        Raises:
            ValueError: If the optimised parameters do not produce a finite cost value.
        """
        if not np.isfinite(self._final_cost[-1]):
            raise ValueError("Optimised parameters do not produce a finite cost value")

    def check_physical_viability(self, x):
        """
        Check if the optimised parameters are physically viable.

        Parameters
        ----------
        x : array-like
            Optimised parameter values.
        """
        if self.cost.problem.model is None:
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
        return (
            f"OptimisationResult:\n"
            f"  Best result from {self.n_runs} run(s).\n"
            f"  Initial parameters: {self.x0}\n"
            f"  Optimised parameters: {self.x}\n"
            f"  Final cost: {self.final_cost}\n"
            f"  Optimisation time: {self.time} seconds\n"
            f"  Number of iterations: {self.n_iterations}\n"
            f"  SciPy result available: {'Yes' if self.scipy_result else 'No'}"
        )

    def average_iterations(self) -> Optional[float]:
        """Calculates the average number of iterations across all runs."""
        return np.mean(self._n_iterations)

    def total_runtime(self) -> Optional[float]:
        """Calculates the total runtime across all runs."""
        return np.sum(self._time)

    @property
    def x(self):
        return self._x[self._best_run] if self._best_run is not None else None

    @property
    def x0(self):
        return self._x0[self._best_run] if self._best_run is not None else None

    @property
    def final_cost(self):
        return self._final_cost[self._best_run] if self._best_run is not None else None

    @property
    def n_iterations(self):
        return (
            self._n_iterations[self._best_run] if self._best_run is not None else None
        )

    @property
    def scipy_result(self):
        return (
            self._scipy_result[self._best_run] if self._best_run is not None else None
        )

    @property
    def time(self):
        return self._time[self._best_run] if self._best_run is not None else None
