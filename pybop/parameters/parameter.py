from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Sequence
from copy import deepcopy
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pybop.parameters.distributions import (
    BaseDistribution,
    JointDistribution,
    Unbounded,
    Uniform,
)
from pybop.parameters.multivariate_distributions import (
    BaseMultivariateDistribution,
    MarginalDistribution,
)
from pybop.transformation.base_transformation import Transformation
from pybop.transformation.transformations import (
    ComposedTransformation,
    IdentityTransformation,
)

# Type aliases
NumericValue = float | int | np.number
ArrayLike = Sequence[NumericValue] | NDArray[np.floating]
BoundsPair = list[float]
Inputs = dict[str, float]


class ParameterError(Exception):
    """Base exception for parameter-related errors."""

    pass


class ParameterValidationError(ParameterError):
    """Raised when parameter validation fails."""

    pass


class ParameterNotFoundError(ParameterError):
    """Raised when a parameter is not found."""

    pass


class Parameter:
    """
    Represents a parameter within the PyBOP framework.

    This class encapsulates the definition of a parameter, including its
    distribution, initial value and transformation.

    Parameters
    ----------
    distribution : BaseDistribution, optional
        Probability distribution for the parameter. If None, an empty
        `pybop.BaseDistribution` will return a NotImplementedError for any
        functionality that requires a distribution, such as `rvs`.
    bounds : tuple[float, float], optional
        Parameter bounds as (lower, upper)
    initial_value : NumericValue, optional
        Initial parameter value
    transformation : Transformation, optional
        Parameter transformation
    """

    def __init__(
        self,
        distribution: BaseDistribution | None = None,
        bounds: BoundsPair | None = None,
        initial_value: float = None,
        transformation: Transformation | None = None,
    ):
        self._distribution = distribution
        self._initial_value = None
        self._transformation = transformation or IdentityTransformation()
        self._transformed_distribution = None

        if bounds is not None:
            if self._distribution is not None:
                raise ParameterError(
                    "Bounds can only be set if no distribution is provided. If a bounded distribution "
                    "is needed, please ensure the distribution itself is bounded."
                )

            # Validate bounds
            if bounds[0] >= bounds[1]:
                raise ParameterValidationError(
                    f"Lower bound ({bounds[0]}) must be less than upper bound ({bounds[1]})"
                )

            # Use a uniform or unbounded distribution to represent the bounds
            if all(np.isfinite(np.asarray(bounds))):
                self._distribution = Uniform(lower=bounds[0], upper=bounds[1])
            else:
                self._distribution = Unbounded(
                    initial_value=initial_value, lower=bounds[0], upper=bounds[1]
                )

        if self._distribution is None:
            if initial_value is not None:
                self._distribution = Unbounded(initial_value=initial_value)
            else:
                self._distribution = BaseDistribution()

        # Set and validate initial value
        self.update_initial_value(value=initial_value)

        # Set the transformed distribution
        self._transformed_distribution = (
            self._distribution.get_transformed_distribution(self._transformation)
        )

    def update_initial_value(self, value: NumericValue | None) -> None:
        """Update the initial parameter value."""
        self._initial_value = float(value) if value is not None else None

        if not self.value_within_bounds(self._initial_value):
            raise ParameterValidationError(
                f"Initial value {value} is outside the parameter bounds {self.bounds}."
            )

    def __repr__(self) -> str:
        """String representation of the parameter."""
        return f"Parameter - Distribution: {self._distribution}, Bounds: {self.bounds}, Initial value: {self._initial_value}"

    def value_within_bounds(self, value: float = None) -> bool:
        """Check if the value is within the bounds."""
        if value is None or self.bounds is None:
            return True

        if self.bounds[0] <= value <= self.bounds[1]:
            return True
        return False

    def get_initial_value(self, transformed: bool = False) -> NDArray | None:
        """Get initial value in either the model space or the transformed search space."""
        if self._initial_value is None and self._distribution is not None:
            # Try to sample from distribution if available
            self.update_initial_value(self._distribution.rvs(1)[0])

        if self._initial_value is None:
            # If still None, just return this
            return None

        if transformed:
            return self._transformation.to_search(self._initial_value)[0]
        return self._initial_value

    def get_mean(self, transformed: bool = False):
        """Get the mean of each parameter, or its initial value."""
        dist = self._transformed_distribution if transformed else self._distribution
        return dist.mean()

    def get_std(self, transformed: bool = False):
        """Get the standard deviation, or an estimate of it."""
        dist = self._transformed_distribution if transformed else self._distribution
        return dist.std()

    def __call__(self, *unused_args, **unused_kwargs) -> float | None:
        """Return the initial value. The unused arguments are to pass pybamm.ParameterValues checks."""
        return self._initial_value

    @property
    def initial_value(self) -> float | None:
        return self._initial_value

    @property
    def bounds(self) -> tuple[float] | None:
        """Parameter bounds as (lower, upper) tuple."""
        lower, upper = self._distribution.support()

        if np.isinf(lower) and np.isinf(upper):
            return None
        else:
            return (lower, upper)

    @property
    def distribution(self) -> BaseDistribution:
        return self._distribution

    @property
    def transformation(self) -> Transformation:
        return self._transformation

    @property
    def transformed_distribution(self) -> BaseDistribution:
        return self._transformed_distribution


class Parameters:
    """
    Container for managing multiple Parameter objects with additional functionality.

    This class provides a comprehensive interface for parameter management including
    validation, transformation, serialisation, and bulk operations.
    """

    def __init__(self, parameters: dict | Parameters | None = None) -> None:
        if not isinstance(parameters, dict | Parameters | None):
            raise TypeError(
                "parameters must be either a dictionary or a pybop.Parameters instance"
            )
        self._parameters = None
        self._distribution = None
        self._transformation = None
        self._transformed_distribution = None

        self._collect_parameters(parameters)

    def _collect_parameters(self, parameters):
        parameters = parameters or {}
        self._parameters = OrderedDict()
        for name, param in parameters.items():
            self.add(name, param, update_attributes=False)
        self._update_attributes()

    def __getitem__(self, name: str) -> Parameter:
        return self.get(name)

    def __setitem__(self, name: str, param: Parameter) -> None:
        self.set(name, param, update_attributes=True)

    def __len__(self) -> int:
        return len(self._parameters)

    def keys(self) -> Iterator[str]:
        """Iterate over parameter names."""
        return iter(self._parameters.keys())

    @property
    def names(self) -> list[str]:
        return list(self._parameters.keys())

    def __iter__(self) -> Iterator[Parameter]:
        return iter(self._parameters.values())

    def _update_attributes(self):
        """
        Method to determine whether to construct a JointDistribution or a MultivariateDistribution
        and to set up the distribution.

        Multivariate distributions are passed to individual parameters via the corresponding
        marginal distribution. The pybop.MarginalDistribution class retains the underlying
        pybop.MultivariateDistribution in the parent_distribution property.
        """
        self._transformation = self.construct_transformation()

        # check if any distribution is a pybop.MarginalDistribution
        multivariate = any(
            isinstance(param.distribution, MarginalDistribution) for param in self
        )

        # if there is a pybop.MarginalDistribution ensure all distributions are marginal
        # distributions of the same parent_distribution
        if multivariate:
            if not all(
                isinstance(param.distribution, MarginalDistribution) for param in self
            ):
                raise TypeError(
                    "A Parameters object with a MarginalDistribution cannot be combined with "
                    "parameters with other types of distributions"
                )
            # Get the parent distribution from the first Parameter object
            parent_dist = next(iter(self)).distribution.parent_distribution
            if not all(
                param.distribution.parent_distribution == parent_dist for param in self
            ):
                raise ValueError(
                    "All MarginalDistributions must share the same parent MultivariateDistribution."
                )
            self._distribution = parent_dist

            # Re-order all properties to match the position of each marginal distribution
            parameter_list = self.names
            index = np.argsort([p.distribution.position for p in self.__iter__()])
            parameter_order = [parameter_list[i] for i in index]
            self._parameters = {key: self._parameters[key] for key in parameter_order}
            self._transformation = ComposedTransformation(
                [self._transformation.transformations[i] for i in index]
            )

        else:
            list_of_distributions = [
                param.distribution for param in self._parameters.values()
            ]
            if len(list_of_distributions) > 0:
                self._distribution = JointDistribution(*list_of_distributions)

        if self._distribution is not None:
            self._transformed_distribution = (
                self._distribution.get_transformed_distribution(self._transformation)
            )

    def add(
        self, name: str, parameter: Parameter, update_attributes: bool = True
    ) -> None:
        """
        Internal method to add a parameter to the collection.

        Parameters
        ----------
        name : str
            Name of the parameter.
        parameter : pybop.Parameter
            The parameter to add.
        update_attributes : bool, optional
            Whether to update the transformation and distributions after adding (default: True).
        """
        if not isinstance(parameter, Parameter):
            raise TypeError("Expected Parameter instance")

        if name in self._parameters:
            raise ParameterError(f"Parameter for '{name}' already exists")

        self._parameters[name] = parameter
        if update_attributes:
            self._update_attributes()

    def join(self, parameters) -> None:
        """
        Join two Parameters objects into the first by copying across each Parameter.

        Parameters
        ----------
        parameters : pybop.Parameters
        """
        for name, param in parameters.items():
            if name not in self._parameters.keys():
                self.add(
                    name, param, update_attributes=False
                )  # don't update every time
            else:
                print(f"Discarding duplicate {name}.")

        self._update_attributes()  # update once when all parameters are added

    def get(self, name: str) -> Parameter:
        """Get a parameter by name."""
        if name not in self._parameters:
            raise ParameterNotFoundError(f"Parameter for '{name}' not found")
        return self._parameters[name]

    def set(self, name: str, param: Parameter, update_attributes: bool = True) -> None:
        """Set a parameter by name."""
        if name not in self._parameters:
            raise ParameterNotFoundError(f"Parameter for '{name}' not found")
        if not isinstance(param, Parameter):
            raise TypeError({"Parameter must be of type pybop.Parameter"})
        self._parameters[name] = param
        if update_attributes:
            self._update_attributes()

    def get_bounds(self, transformed: bool = False) -> dict:
        """
        Get bounds for each parameter as a dictionary.

        Parameters
        ----------
        transformed : bool
            If True, the transformation is applied to the output (default: False).
        """
        bounds_array = self.get_bounds_array(transformed=transformed).T

        return {"lower": bounds_array[0], "upper": bounds_array[1]}

    def get_bounds_array(self, transformed: bool = False) -> np.ndarray:
        """
        Retrieve parameter bounds in numpy format.

        Returns
        -------
        bounds : numpy.ndarray
            An array of shape (n_parameters, 2) containing the bounds for each parameter.
        """
        dist = self._transformed_distribution if transformed else self._distribution
        return dist.support().T

    def update(
        self,
        initial_values: ArrayLike | Inputs | None = None,
        **individual_updates: dict[str, Any],
    ) -> None:
        """
        Update multiple parameters efficiently.

        Parameters
        ----------
        initial_values : array-like or dict, optional
            New initial values (by position or name)
        bounds : sequence or dict, optional
            New bounds (by position or name)
        **individual_updates : dict
            Individual parameter updates with parameter names as keys
        """
        # Handle individual parameter updates
        for param_name, updates in individual_updates.items():
            param = self.get(param_name)  # Raises if not found

            if isinstance(updates, dict):
                if "initial_value" in updates:
                    param.update_initial_value(updates["initial_value"])

        # Handle bulk updates
        if initial_values is not None:
            self._bulk_update_initial_values(initial_values)

    def _bulk_update_initial_values(self, values: ArrayLike | Inputs) -> None:
        """Update initial values in bulk."""
        if isinstance(values, dict):
            for name, value in values.items():
                self.get(name).update_initial_value(value)
        else:
            values_array = np.atleast_1d(values)
            param_list = list(self._parameters.values())

            if len(values_array) != len(param_list):
                raise ParameterValidationError(
                    f"Values array length {len(values_array)} doesn't match "
                    f"parameter count {len(param_list)}"
                )

            for param, value in zip(param_list, values_array, strict=False):
                param.update_initial_value(value)

    def sample_from_distribution(
        self,
        n_samples: int = 1,
        random_state: int | None = None,
        transformed: bool = False,
    ) -> NDArray[np.floating] | None:
        """
        Sample from a joint or multivariate distribution.

        Parameters
        ----------
        n_samples : int
            The number of samples to draw (default: 1).
        random_state : int, optional
            The random state seed for reproducibility (default: None).
        transformed: bool, optional
            If True, the transformation is applied to the output (default: False).

        Returns
        -------
        NDArray[np.floating] or None
            Array of shape (n_samples, n_parameters) or None if any distribution is missing
        """
        dist = self._transformed_distribution if transformed else self._distribution

        samples = dist.rvs(n_samples, random_state=random_state)
        return np.atleast_2d(samples)

    def get_mean(self, transformed: bool = False):
        """
        Get the mean of each parameter, or its initial value.

        Parameters
        ----------
        transformed : bool, optional
            If True, the transformation is applied to the output (default: False).
        """
        dist = self._transformed_distribution if transformed else self._distribution
        return dist.mean()

    def get_std(self, transformed: bool = False) -> np.ndarray:
        """
        Get the standard deviation, or an estimate of it, for each parameter.

        Parameters
        ----------
        transformed : bool, optional
            If True, the transformation is applied to the output (default: False).
        """
        dist = self._transformed_distribution if transformed else self._distribution
        return dist.std()

    def get_covariance(self, transformed: bool = False):
        """
        Get the covariance matrix, or an estimate of it.

        Parameters
        ----------
        transformed : bool, optional
            If True, the transformation is applied to the output (default: False).
        """
        dist = self._transformed_distribution if transformed else self._distribution
        return dist.cov()

    @property
    def distribution(
        self,
    ) -> BaseMultivariateDistribution | JointDistribution | None:
        """Return the joint or multivariate distribution."""
        return self._distribution

    def get_initial_values(self, transformed: bool = False) -> NDArray[np.floating]:
        """
        Get initial values as array.

        Parameters
        ----------
        transformed : bool, default=False
            Whether to apply transformations to bounds

        Returns
        -------
        NDArray[np.floating]
            Array of initial values
        """
        values = []
        for name, param in self._parameters.items():
            value = param.get_initial_value(transformed=transformed)
            if value is None:
                raise ParameterError(f"Parameter '{name}' has no initial value")

            values.append(value)

        return np.asarray(values)

    @property
    def transformation(self) -> Transformation:
        """Get the transformation for the parameters."""
        return self._transformation

    def construct_transformation(self) -> Transformation:
        """
        Create a ComposedTransformation object from the individual parameter transformations.
        """
        transformations = []

        for param in self._parameters.values():
            transformations.append(param.transformation)

        if transformations == []:
            return None

        return ComposedTransformation(transformations)

    def get_bounds_for_plotly(self, transformed: bool = False) -> np.ndarray:
        """
        Retrieve parameter bounds in the format expected by Plotly.

        Returns
        -------
        bounds : numpy.ndarray
            An array of shape (n_parameters, 2) containing the bounds for each parameter.
        """
        bounds_array = self.get_bounds_array(transformed=transformed)

        # Validate that all parameters have bounds
        if not np.isfinite(bounds_array).all():
            raise ValueError("All parameters require bounds for plot.")

        return bounds_array

    def to_dict(self, values: str | ArrayLike | None = None) -> Inputs:
        """
        Return values as a dictionary of inputs.

        Parameters
        ----------
        values : str or array-like, optional
            Which values to use ('initial') or custom array. Default is "initial".

        Returns
        -------
        Inputs
            Dictionary mapping parameter names to values
        """
        if values is None:
            values = "initial"
        params = self._parameters.items()

        if isinstance(values, str) and values == "initial":
            return {name: param.get_initial_value() for name, param in params}
        else:
            # Custom values array
            values_array = np.atleast_1d(values)
            if len(values_array) != len(self._parameters):
                raise ParameterValidationError(
                    f"Values array length {len(values_array)} doesn't match parameter count {len(self._parameters)}"
                )
            return dict(zip(self._parameters.keys(), values_array, strict=False))

    def verify_inputs(self, inputs: Inputs) -> bool:
        """Check if the inputs are valid parameters."""
        valid = True
        for name, param in self._parameters.items():
            if param.bounds is not None:
                if not param.value_within_bounds(inputs[name]):
                    valid = False
        return valid

    def __repr__(self) -> str:
        param_summary = "\n".join(
            f" {name}: {param}" for name, param in self._parameters.items()
        )
        return f"Parameters({len(self)}):\n{param_summary}"

    def to_inputs_list(self, values: np.ndarray | list[np.ndarray]) -> list[Inputs]:
        """
        Return parameter values as a list of dictionaries, as required for multiprocessing.
        """
        values = np.asarray(values)
        if values.ndim == 1:
            return [self.to_dict(values=values)]

        inputs_list = []
        for val in values:
            inputs_list.append(self.to_dict(values=val))
        return inputs_list

    def convert_grad_to_array(self, grad: dict[str, np.ndarray]) -> np.ndarray:
        """Get an array of sensitivities with the parameters in the expected order."""
        return np.vstack([grad[key] for key in self.names]).T

    def copy(self) -> Parameters:
        """Create a deep copy of the Parameters object."""
        return deepcopy(self)

    def __contains__(self, name: str) -> bool:
        return name in self._parameters

    def values(self) -> Iterator[Parameter]:
        """Iterate over parameters."""
        return iter(self._parameters.values())

    def items(self) -> Iterator[tuple[str, Parameter]]:
        """Iterate over (name, parameter) pairs."""
        return iter(self._parameters.items())
