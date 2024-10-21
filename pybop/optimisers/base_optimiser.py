import warnings
from typing import Optional, Union

import numpy as np
from scipy.optimize import OptimizeResult

from pybop import (
    BaseCost,
    BaseLikelihood,
    DesignCost,
    Inputs,
    Parameter,
    Parameters,
    WeightedCost,
)


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
    minimising : bool, optional
        If True, the target is to minimise the cost, else target is to maximise by minimising
        the negative cost (default: True).
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
        self.x0 = None
        self.bounds = None
        self.sigma0 = 0.02
        self.verbose = True
        self.log = dict(x=[], x_best=[], cost=[], cost_best=[])
        self.minimising = True
        self._transformation = None
        self._needs_sensitivities = False
        self.physical_viability = False
        self.allow_infeasible_solutions = False
        self.default_max_iterations = 1000
        self.result = None

        if isinstance(cost, BaseCost):
            self.cost = cost
            self.parameters = self.cost.parameters
            self._transformation = self.cost.transformation
            self.set_allow_infeasible_solutions()
            if isinstance(cost, WeightedCost):
                self.minimising = cost.minimising
            if isinstance(cost, (BaseLikelihood, DesignCost)):
                self.minimising = False

        else:
            try:
                self.x0 = optimiser_kwargs.get("x0", [])
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
                self.minimising = True

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
        self.minimising = self.unset_options.pop("minimising", self.minimising)
        if "allow_infeasible_solutions" in self.unset_options.keys():
            self.set_allow_infeasible_solutions(
                self.unset_options.pop("allow_infeasible_solutions")
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
        self.result = self._run()

        if self.verbose:
            print(self.result)

        # Store the optimised parameters
        self.parameters.update(values=self.result.x)

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

    def log_update(self, x=None, x_best=None, cost=None, cost_best=None):
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
            x = apply_transformation(x)
            self.log["x"].extend(x)

        if x_best is not None:
            x_best = convert_to_list(x_best)
            x_best = apply_transformation(x_best)
            self.log["x_best"].extend([x_best])

        if cost is not None:
            cost = convert_to_list(cost)
            self.log["cost"].extend(cost)

        if cost_best is not None:
            cost_best = convert_to_list(cost_best)
            self.log["cost_best"].extend(cost_best)

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


class OptimisationResult:
    """
    Stores the result of the optimisation.

    Attributes
    ----------
    x : ndarray
        The solution of the optimisation.
    final_cost : float
        The cost associated with the solution x.
    nit : int
        Number of iterations performed by the optimiser.
    scipy_result : scipy.optimize.OptimizeResult, optional
        The result obtained from a SciPy optimiser.
    """

    def __init__(
        self,
        x: Union[Inputs, np.ndarray] = None,
        cost: Union[BaseCost, None] = None,
        final_cost: Optional[float] = None,
        n_iterations: Optional[int] = None,
        optim: Optional[BaseOptimiser] = None,
        time: Optional[float] = None,
        scipy_result=None,
    ):
        self.x = x
        self.cost = cost
        self.final_cost = (
            final_cost if final_cost is not None else self._calculate_final_cost()
        )
        self.n_iterations = n_iterations
        self.scipy_result = scipy_result
        self.optim = optim
        self.time = time
        if isinstance(self.optim, BaseOptimiser):
            self.x0 = self.optim.parameters.initial_value()
        else:
            self.x0 = None

        # Check that the parameters produce finite cost, and are physically viable
        self._validate_parameters()
        self.check_physical_viability(self.x)

    def _calculate_final_cost(self) -> float:
        """
        Calculate the final cost using the cost function and optimised parameters.

        Returns:
            float: The calculated final cost.
        """
        return self.cost(self.x)

    def get_scipy_result(self) -> Optional[OptimizeResult]:
        """
        Get the SciPy optimisation result object.

        Returns:
            OptimizeResult or None: The SciPy optimisation result object if available, None otherwise.
        """
        return self.scipy_result

    def _validate_parameters(self) -> None:
        """
        Validate the optimised parameters and ensure they produce a finite cost value.

        Raises:
            ValueError: If the optimized parameters do not produce a finite cost value.
        """
        if not np.isfinite(self.final_cost):
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
            f"  Initial parameters: {self.x0}\n"
            f"  Optimised parameters: {self.x}\n"
            f"  Final cost: {self.final_cost}\n"
            f"  Optimisation time: {self.time} seconds\n"
            f"  Number of iterations: {self.n_iterations}\n"
            f"  SciPy result available: {'Yes' if self.scipy_result else 'No'}"
        )
