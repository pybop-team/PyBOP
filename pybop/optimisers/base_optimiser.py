import warnings
from copy import deepcopy
from typing import Union

import jax.numpy as jnp
import numpy as np

from pybop import BaseCost, Inputs, OptimisationResult, Parameter, Parameters


class BaseOptimiser:
    """
    A base class for defining optimisation methods.

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
            Initial values of the (search) parameters for the optimisation.
        bounds : dict
            Dictionary containing bounds for the (search) parameters with keys 'lower' and 'upper'.
        sigma0 : float or sequence
            Initial step size or standard deviation in the (search) parameters. Either a scalar value
            (same for all coordinates) or an array with one entry per dimension.
            Not all methods will use this information.
        verbose : bool, optional
            If True, the optimisation progress is printed (default: False).
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
        self.log = dict(x=[], x_best=[], x_search=[], cost=[], cost_best=[])
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
        # Set initial values, if x0 is None, initial values are unmodified
        x0_search = self.unset_options.pop("x0", None)
        if x0_search is not None:
            x0_model = self.apply_transformation(x0_search)
            self.parameters.update(initial_values=x0_model)
        self.x0 = self.parameters.reset_initial_value(apply_transform=True)

        # Set the search bounds (for all or no parameters)
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

    def call_cost(
        self,
        x: Union[Inputs, list],
        cost: Union[BaseCost, callable],
        calculate_grad: bool = False,
        apply_transform: bool = True,
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
        return cost(
            x,
            calculate_grad=calculate_grad,
            apply_transform=apply_transform,
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
            if isinstance(array_like, (list, tuple, np.ndarray, jnp.ndarray)):
                return list(array_like)
            elif isinstance(array_like, (int, float)):
                return [array_like]
            else:
                raise TypeError("Input must be a list, tuple, or numpy array")

        if x is not None:
            x = convert_to_list(x)
            self.log["x_search"].extend(x)
            x = self.apply_transformation(x)
            self.log["x"].extend(x)

        if x_best is not None:
            x_best = self.apply_transformation([x_best])
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

    def apply_transformation(self, values):
        """Apply transformation if it exists."""
        if self._transformation:
            return [self._transformation.to_model(value) for value in values]
        return values

    @property
    def needs_sensitivities(self):
        return self._needs_sensitivities

    @property
    def transformation(self):
        return self._transformation

    @property
    def minimising(self):
        return self._minimising
