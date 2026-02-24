from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.stats as stats
from numpy.typing import NDArray

from pybop.parameters.distributions import Distribution, JointDistribution
from pybop.parameters.multivariate_distributions import (
    BaseMultivariateDistribution,
    MarginalDistribution,
)
from pybop.transformation.base_transformation import Transformation
from pybop.transformation.transformations import (
    ComposedTransformation,
    IdentityTransformation,
    LogTransformation,
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


@dataclass(frozen=True)
class Bounds:
    """
    Immutable bounds representation with validation.

    Attributes
    ----------
    lower : float
        Lower bound (inclusive)
    upper : float
        Upper bound (inclusive)
    """

    lower: float
    upper: float

    def __post_init__(self) -> None:
        if self.lower >= self.upper:
            raise ParameterValidationError(
                f"Lower bound ({self.lower}) must be less than upper bound ({self.upper})"
            )

    def contains(self, value: NumericValue) -> bool:
        """Check if value is within bounds."""
        return self.lower <= value <= self.upper

    def contains_array(self, values: ArrayLike) -> bool:
        """Check if all values in array are within bounds."""
        arr = np.asarray(values)
        return bool(np.all((arr >= self.lower) & (arr <= self.upper)))

    def clip(self, value: NumericValue) -> float:
        """Clip value to bounds."""
        return float(np.clip(value, self.lower, self.upper))

    def clip_array(self, values: ArrayLike) -> NDArray[np.floating]:
        """Clip array values to bounds."""
        return np.clip(values, self.lower, self.upper)

    def width(self) -> float:
        """Return the width of the bounds."""
        return self.upper - self.lower


class Parameter:
    """
    Represents a parameter within the PyBOP framework.

    This class encapsulates the definition of a parameter, including its
    initial value, bounds.

    Parameters
    ----------
    distribution : stats.distribution.rv_frozen | Distribution, optional
        Distribution of the parameter
    bounds : tuple[float, float], optional
        Parameter bounds as (lower, upper)
    initial_value : NumericValue, optional
        Initial parameter value
    transformation : Transformation, optional
        Parameter transformation
    """

    def __init__(
        self,
        distribution: stats.distributions.rv_frozen | Distribution | None = None,
        bounds: BoundsPair | None = None,
        initial_value: float = None,
        transformation: Transformation | None = None,
    ) -> None:
        self._distribution = distribution
        self._bounds = None
        self._initial_value = None
        self._transformation = transformation or IdentityTransformation()

        # Some optimisers (EP-BOLFI) and all samplers require distribution properties in the search space
        # rather than the model space. Some transformations are not suitable for some multivariate
        # distributions as they are currently implemented in this context
        self._check_compatible_transformation()

        if self._distribution is not None:
            lower, upper = self._distribution.support()
            if np.isinf(lower) and np.isinf(upper):
                self._bounds = None
            else:
                self._bounds = Bounds(float(lower), float(upper))

        if bounds is not None:
            if distribution is not None:
                raise ParameterError(
                    "Bounds can only be set if no distribution is provided. If a bounded distribution "
                    "is needed, please ensure the distribution itself is bounded."
                )
            # Set bounds with validation
            self._bounds = Bounds(float(bounds[0]), float(bounds[1]))
            # Add uniform distribution for finite bounds in order to sample initial values
            if all(np.isfinite(np.asarray(bounds))):
                self._distribution = stats.uniform(
                    loc=bounds[0], scale=bounds[1] - bounds[0]
                )

        # Set and validate initial value
        self.update_initial_value(value=initial_value)

    def _check_compatible_transformation(self) -> None:
        """Raise an error if the transformation is not compatible with the distribution."""
        if isinstance(self._distribution, MarginalDistribution):
            allowed_transformations = (
                self._distribution.parent_distribution.compatible_transformations
            )

            if not isinstance(self._transformation, allowed_transformations):
                raise TypeError(
                    "The transformation provided is not compatible with "
                    f"pybop.{self._distribution.parent_distribution.name}. Only "
                    + ", ".join([trans.__name__ for trans in allowed_transformations])
                    + " are allowed."
                )

    def sample_from_distribution(
        self,
        n_samples: int = 1,
        *,
        random_state: int | None = None,
        transformed: bool = False,
    ) -> NDArray[np.floating] | None:
        """
        Sample from parameter's distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw (default: 1).
        random_state : int, optional
            Random seed for reproducibility.
        transformed : bool, optional
            Whether to apply transformation to samples (default: False).

        Returns
        -------
        NDArray[np.floating] or None
            Array of samples, or None if no distribution exists exists
        """
        if self._distribution is None:
            return None

        samples = self._distribution.rvs(n_samples, random_state=random_state)
        samples = np.atleast_1d(samples).astype(float)

        if transformed:
            samples = np.array([self._transformation.to_search(s)[0] for s in samples])

        return samples

    def update_initial_value(self, value: NumericValue | None) -> None:
        """Update the initial parameter value."""
        self._initial_value = float(value) if value is not None else None
        self._validate_initial_value_within_bounds()

    def __repr__(self) -> str:
        """String representation of the parameter."""
        return f"Parameter - Distribution: {self._distribution}, Bounds: {self.bounds}, Initial value: {self._initial_value}"

    def _validate_initial_value_within_bounds(self) -> None:
        """Validate that the initial value is within the bounds."""
        if self._initial_value is not None:
            if self._bounds is None:
                return

            if not self._bounds.contains(self._initial_value):
                raise ParameterValidationError(
                    f"Initial value {self._initial_value} is outside bounds {self.bounds}"
                )

    def get_initial_value(self, transformed: bool = False) -> NDArray | None:
        """Get initial value in either the model space or the transformed search space."""
        if self._initial_value is None and self._distribution is not None:
            # Try to sample from distribution if available
            self.update_initial_value(self.sample_from_distribution(1)[0])

        if self._initial_value is None:
            # If still None, just return this
            return None

        if transformed:
            return self._transformation.to_search(self._initial_value)[0]
        return self._initial_value

    def get_mean(self, transformed: bool = False):
        """Get the mean of each parameter, or its initial value."""
        if self.distribution is not None:
            mean = self.distribution.mean()
        elif self.bounds is not None and np.isfinite(self.bounds[1] - self.bounds[0]):
            lower, upper = self.bounds
            mean = (lower + upper) / 2
        else:
            mean = self.get_initial_value()

        if transformed and mean is not None:
            mean = self.transformation.to_search(np.asarray(mean))

        return np.asarray(mean).item()

    def get_std(self, transformed: bool = False):
        """Get the standard deviation, or an estimate of it."""
        if self.distribution is not None:
            std = self.distribution.std()
        elif self.bounds is not None and np.isfinite(self.bounds[1] - self.bounds[0]):
            lower, upper = self.bounds
            std = 0.05 * (upper - lower)
        else:
            std = 0.05 * self.get_initial_value()

        if transformed and std is not None:
            std = self.transformation.convert_standard_deviation(
                std, self.get_mean(transformed=True)
            )

        return np.asarray(std).item()

    def __call__(self, *unused_args, **unused_kwargs) -> float | None:
        """Return the initial value. The unused arguments are to pass pybamm.ParameterValues checks."""
        return self._initial_value

    @property
    def initial_value(self) -> float | None:
        return self._initial_value

    @property
    def bounds(self) -> tuple | None:
        """Parameter bounds as (lower, upper) tuple."""
        return (
            None if self._bounds is None else (self._bounds.lower, self._bounds.upper)
        )

    @property
    def distribution(self) -> Any | None:
        return self._distribution

    @property
    def transformation(self) -> Transformation:
        return self._transformation


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
        self._multivariate = None
        self._transform = None

        self._collect_parameters(parameters)
        self._transform = self.construct_transformation()
        self.update_distribution()

    def _collect_parameters(self, parameters):
        parameters = parameters or {}
        self._parameters = OrderedDict()
        for name, param in parameters.items():
            self._add(name, param, update_transform=False, update_distribution=False)

    def __getitem__(self, name: str) -> Parameter:
        return self.get(name)

    def __setitem__(self, name: str, param: Parameter) -> None:
        self.set(name, param)

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

    def add(
        self, name: str, parameter: Parameter, update_distribution: bool = True
    ) -> None:
        """Add a parameter to the collection."""
        self._add(name, parameter, update_distribution=update_distribution)

    def update_distribution(self):
        """
        Method to determine whether to construct a JointDistribution or a MultivariateDistribution
        and to set up the distribution.

        Multivariate distributions are passed to individual parameters via the corresponding
        marginal distribution. The pybop.MarginalDistribution class retains the underlying
        pybop.MultivariateDistribution in the parent_distribution property.
        """
        # check if any distribution is a pybop.MarginalDistribution
        self._multivariate = any(
            isinstance(param.distribution, MarginalDistribution) for param in self
        )

        # if there is a pybop.MarginalDistribution ensure all distributions are marginal
        # distributions of the same parent_distribution
        if self._multivariate:
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
            self._transform._transformations = [  # noqa: SLF001
                self._transform.transformations[i] for i in index
            ]

        else:
            list_of_distributions = [
                param.distribution
                for param in self._parameters.values()
                if param.distribution is not None
            ]
            if len(list_of_distributions) == len(self):
                self._distribution = JointDistribution(*list_of_distributions)
            else:
                self._distribution = None

    def _add(
        self,
        name: str,
        parameter: Parameter,
        update_transform: bool = True,
        update_distribution: bool = True,
    ) -> None:
        """
        Internal method to add a parameter to the collection.

        Parameters
        ----------
        name : str
            Name of the parameter.
        parameter : pybop.Parameter
            The parameter to add.
        update_transform : bool, optional
            Whether to update the transformation after adding (default: True).
        update_distribution : bool, optional
            Whether to update the joint or multivariate distribution (default: True).
        """
        if not isinstance(parameter, Parameter):
            raise TypeError("Expected Parameter instance")

        if name in self._parameters:
            raise ParameterError(f"Parameter for '{name}' already exists")

        self._parameters[name] = parameter

        if update_transform:
            self._transform = self.construct_transformation()
        if update_distribution:
            self.update_distribution()

    def join(self, parameters=None):
        """
        Join two Parameters objects into the first by copying across each Parameter.

        Parameters
        ----------
        parameters : pybop.Parameters
        """
        for name, param in parameters.items():
            if name not in self._parameters.keys():
                self.add(
                    name, param, update_distribution=False
                )  # don't update every time
            else:
                print(f"Discarding duplicate {name}.")

        self.update_distribution()  # update once when all parameters are added

    def get(self, name: str) -> Parameter:
        """Get a parameter by name."""
        if name not in self._parameters:
            raise ParameterNotFoundError(f"Parameter for '{name}' not found")
        return self._parameters[name]

    def set(
        self, name: str, param: Parameter, update_distribution: bool = True
    ) -> None:
        """Set a parameter by name."""
        if name not in self._parameters:
            raise ParameterNotFoundError(f"Parameter for '{name}' not found")
        if not isinstance(param, Parameter):
            raise TypeError({"Parameter must be of type pybop.Parameter"})
        self._parameters[name] = param
        if update_distribution:
            self.update_distribution()

    def get_bounds(self, transformed: bool = False) -> dict:
        """
        Get bounds for each parameter as a dictionary.

        Parameters
        ----------
        transformed : bool
            If True, the transformation is applied to the output (default: False).
        """
        bounds = {"lower": [], "upper": []}
        for param in self._parameters.values():
            lower, upper = param.bounds or (-np.inf, np.inf)

            if transformed and param.bounds is not None:
                if isinstance(param.transformation, LogTransformation) and lower == 0:
                    bound_one = -np.inf
                else:
                    bound_one = float(param.transformation.to_search(lower)[0])
                bound_two = float(param.transformation.to_search(upper)[0])

                if np.isnan(bound_one) or np.isnan(bound_two):
                    raise ValueError("Transformed bounds resulted in NaN values.")

                lower = np.minimum(bound_one, bound_two)
                upper = np.maximum(bound_one, bound_two)

            bounds["lower"].append(lower)
            bounds["upper"].append(upper)

        return bounds

    def get_bounds_array(self, transformed: bool = False) -> np.ndarray:
        """
        Retrieve parameter bounds in numpy format.

        Returns
        -------
        bounds : numpy.ndarray
            An array of shape (n_parameters, 2) containing the bounds for each parameter.
        """
        bounds = self.get_bounds(transformed=transformed)
        return np.column_stack([bounds["lower"], bounds["upper"]])

    def update(
        self,
        *,
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
        *,
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
        if self._distribution is None:
            return None

        samples = self._distribution.rvs(n_samples, random_state=random_state)
        samples = np.atleast_2d(samples)

        if transformed:
            samples = np.asarray([self.transformation.to_search(s) for s in samples])

        return samples

    def get_mean(self, transformed: bool = False):
        """
        Get the mean of each parameter, or its initial value.

        Parameters
        ----------
        transformed : bool, optional
            If True, the transformation is applied to the output (default: False).
        """
        if self._multivariate:
            if transformed:
                return self.transformed_distribution_properties["mean"]
            else:
                return self.distribution.properties["mean"]

        else:
            means = []
            for param in self._parameters.values():
                means.append(param.get_mean(transformed=transformed))
            return np.asarray(means).T

    def get_std(self, transformed: bool = False) -> list:
        """
        Get the standard deviation, or an estimate of it, for each parameter.

        Parameters
        ----------
        transformed : bool, optional
            If True, the transformation is applied to the output (default: False).
        """
        standard_deviations = []

        for param in self._parameters.values():
            standard_deviations.append(param.get_std(transformed=transformed))
        return standard_deviations

    def get_covariance(self, transformed: bool = False):
        """
        Get the covariance matrix, or an estimate of it.

        Parameters
        ----------
        transformed : bool, optional
            If True, the transformation is applied to the output (default: False).
        """
        if self._multivariate:
            if transformed:
                return self.transformed_distribution_properties["cov"]
            else:
                return self.distribution.properties["cov"]

        else:
            standard_deviations = self.get_std(transformed=transformed)
            return (np.eye(len(self)) * np.asarray(standard_deviations)) ** 2

    @property
    def distribution(
        self,
    ) -> BaseMultivariateDistribution | JointDistribution | None:
        """Return the joint or multivariate distribution."""
        return self._distribution

    def get_initial_values(self, *, transformed: bool = False) -> NDArray[np.floating]:
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
        return self._transform

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

    @property
    def transformed_distribution_properties(self):
        # retrieve properties of the distribution in the search space
        # needed for ep-bolfi optimiser
        if self._multivariate:
            return self.distribution.transformed_properties(self._transform)
        else:
            raise NotImplementedError

    def get_bounds_for_plotly(self, transformed: bool = False) -> np.ndarray:
        """
        Retrieve parameter bounds in the format expected by Plotly.

        Returns
        -------
        bounds : numpy.ndarray
            An array of shape (n_parameters, 2) containing the bounds for each parameter.
        """
        bounds = self.get_bounds(transformed=transformed)

        # Validate that all parameters have bounds
        if not np.isfinite(list(bounds.values())).all():
            raise ValueError("All parameters require bounds for plot.")

        return np.asarray(list(bounds.values())).T

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
                if not param._bounds.contains(inputs[name]):  # noqa: SLF001
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
