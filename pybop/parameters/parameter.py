from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.stats as stats
from numpy.typing import NDArray

from pybop.parameters.distributions import Distribution
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
    distribution : stats.distribution.rv_frozen | Distribution
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
        distribution: stats.rv_continuous
        | Distribution
        | stats._distribution_infrastructure.ContinuousDistribution
        | None = None,
        distribution_params: dict | None = None,
        bounds: BoundsPair | None = None,
        initial_value: float = None,
        transformation: Transformation | None = None,
    ) -> None:
        self._distribution = distribution
        self._bounds = None
        self._transformation = transformation or IdentityTransformation()

        if isinstance(distribution, stats.rv_continuous):
            distribution = stats.make_distribution(distribution)
            if "loc" in distribution_params:
                loc = distribution_params["loc"]
                del distribution_params["loc"]
            else:
                loc = 0.0
            if "scale" in distribution_params:
                scale = distribution_params["scale"]
                del distribution_params["scale"]
            else:
                scale = 0.0
            X = distribution(**distribution_params)
            self._distribution = scale * X + loc
        elif (
            isinstance(
                distribution,
                (
                    Distribution,
                    stats._distribution_infrastructure.ContinuousDistribution,  # noqa SLF001
                ),
            )
            or distribution is None
        ):
            self._distribution = distribution
        else:
            raise TypeError(
                "The distribution must be of type pybop.Distribution, stats.rv_continous, or stats._distribution_infrastructure.ContinousDistribution"
            )

        if self._distribution is not None:
            lower, upper = self._distribution.support()
            if np.isinf(lower) and np.isinf(upper):
                self._bounds = None
            else:
                self._bounds = Bounds(lower, upper)

        if bounds is not None:
            if distribution is not None:
                raise ParameterError(
                    "Bounds can only be set if no distribution is provided. If a bounded distribution is needed, please ensure the distribution itself is bounded."
                )
            # Set bounds with validation
            self._bounds = Bounds(bounds[0], bounds[1])
            # Add uniform distribution for finite bounds in order to sample initial values
            if all(np.isfinite(np.asarray(bounds))):
                self._distribution = stats.Uniform(a=bounds[0], b=bounds[1])

        if initial_value is None and self._distribution is not None:
            initial_value = self.sample_from_distribution()[0]

        # Validate and set values
        self._initial_value = (
            float(initial_value) if initial_value is not None else None
        )

        # Validate initial values are within bounds
        self._validate_values_within_bounds()

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
        transformed : bool
            Whether to apply transformation to samples (default: False).

        Returns
        -------
        NDArray[np.floating] or None
            Array of samples, or None if no distribution exists exists
        """
        if self._distribution is None:
            return None

        samples = self._distribution.sample(n_samples, rng=random_state)
        samples = np.atleast_1d(samples).astype(float)

        if transformed:
            samples = np.array([self._transformation.to_search(s)[0] for s in samples])

        return samples

    def update_initial_value(self, value: NumericValue) -> None:
        """
        Update the initial parameter value.

        Parameters
        ----------
        value : NumericValue
            New initial value
        """
        self._initial_value = float(value)

    def __repr__(self) -> str:
        """String representation of the parameter."""
        return f"Parameter - Distribution: {self._distribution}, Bounds: ({self.bounds[0]}, {self.bounds[1]}), Initial value: {self.initial_value}"

    def _validate_values_within_bounds(self) -> None:
        """Validate that initial values are within bounds."""
        if self._bounds is None or self._initial_value is None:
            return

        if not self._bounds.contains(self._initial_value):
            raise ParameterValidationError(
                f"Initial value {self._initial_value} is outside bounds {self.bounds}"
            )

    def get_initial_value_transformed(self) -> NDArray | None:
        """Get initial value in transformed space."""
        if self._initial_value is None:
            return None
        return self._transformation.to_search(self._initial_value)[0]

    def __call__(self, *unused_args, **unused_kwargs) -> float:
        "Return the initial value. The unused arguments are to pass pybamm.ParameterValues checks."
        return self._initial_value

    @property
    def initial_value(self) -> float:
        return self._initial_value

    @property
    def bounds(self) -> BoundsPair | None:
        """Parameter bounds as (lower, upper) tuple."""
        return (
            None if self._bounds is None else [self._bounds.lower, self._bounds.upper]
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

    def __init__(self, parameters: dict | Parameters = None) -> None:
        if parameters is None:
            parameters = {}
        elif not isinstance(parameters, (dict, Parameters)):
            raise TypeError(
                "parameters must be either a dictionary or a pybop.Parameters instance"
            )

        self._parameters = OrderedDict()
        for name, param in parameters.items():
            self._add(name, param, update_transform=False)

        self._transform = self.construct_transformation()

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

    def add(self, name: str, parameter: Parameter) -> None:
        """Add a parameter to the collection."""
        self._add(name, parameter)

    def _add(
        self, name: str, parameter: Parameter, update_transform: bool = True
    ) -> None:
        """
        Internal method to add a parameter to the collection.

        Parameters
        ----------
        parameter : Parameter
            Parameter to add
        update_transform : bool, optional
            Whether to update the transformation after adding (default: True)
        """
        if not isinstance(parameter, Parameter):
            raise TypeError("Expected Parameter instance")

        if name in self._parameters:
            raise ParameterError(f"Parameter for '{name}' already exists")

        self._parameters[name] = parameter

        if update_transform:
            self._transform = self.construct_transformation()

    def remove(self, name: str) -> Parameter:
        """Remove parameter and return it."""
        if not isinstance(name, str):
            raise TypeError("The input name is not a string.")
        if name not in self._parameters:
            raise ParameterNotFoundError(f"Parameter for '{name}' not found")
        return self._parameters.pop(name)

    def join(self, parameters=None):
        """
        Join two Parameters objects into the first by copying across each Parameter.

        Parameters
        ----------
        parameters : pybop.Parameters
        """
        for name, param in parameters.items():
            if name not in self._parameters.keys():
                self.add(name, param)
            else:
                print(f"Discarding duplicate {name}.")

    def get(self, name: str) -> Parameter:
        """Get a parameter by name."""
        if name not in self._parameters:
            raise ParameterNotFoundError(f"Parameter for '{name}' not found")
        return self._parameters[name]

    def set(self, name: str, param: Parameter) -> None:
        """Get a parameter by name."""
        if name not in self._parameters:
            raise ParameterNotFoundError(f"Parameter for '{name}' not found")
        if not isinstance(param, Parameter):
            raise TypeError({"Paremeter must be of type pybop.ParemterInfo"})
        self._parameters[name] = param

    def get_bounds(self, transformed: bool = False) -> dict:
        """
        Get bounds, for either all or no parameters.

        Parameters
        ----------
        transformed : bool
            If True, the transformation is applied to the output (default: False).
        """
        bounds = {"lower": [], "upper": []}
        for param in self._parameters.values():
            lower, upper = param.bounds or (-np.inf, np.inf)

            if (
                transformed
                and param.bounds is not None
                and param.transformation is not None
            ):
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

    def sample_from_distributions(
        self,
        n_samples: int = 1,
        *,
        random_state: int | None = None,
        transformed: bool = False,
    ) -> NDArray[np.floating] | None:
        """
        Sample from all parameter distributions.

        Returns
        -------
        NDArray[np.floating] or None
            Array of shape (n_samples, n_parameters) or None if any distribution is missing
        """
        all_samples = []

        for param in self._parameters.values():
            samples = param.sample_from_distribution(
                n_samples, random_state=random_state, transformed=transformed
            )
            if samples is None:
                return None
            all_samples.append(samples)

        return np.column_stack(all_samples)

    def get_sigma0(self, transformed: bool = False) -> list:
        """
        Get the standard deviation, for either all or no parameters.

        Parameters
        ----------
        transformed : bool
            If True, the transformation is applied to the output (default: False).
        """
        sigma0 = []

        for param in self._parameters.values():
            sig = None
            if param.distribution is not None:
                sig = param.distribution.standard_deviation()
            elif param.bounds is not None:
                lower, upper = param.bounds
                if np.isfinite(upper - lower):
                    sig = 0.05 * (upper - lower)

            if transformed and sig is not None and param.transformation is not None:
                sig = np.ndarray.item(
                    param.transformation.convert_standard_deviation(
                        sig, param.transformation.to_search(param.initial_value)[0]
                    )
                )

            sigma0.extend([sig or 0.05])
        return sigma0

    def distributions(self) -> list:
        """Return the initial distribution of each parameter."""
        return [
            param.distribution
            for param in self._parameters.values()
            if param.distribution is not None
        ]

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
            value = param.initial_value
            if value is None:
                # Try to sample from distribution if available
                if param.distribution is not None:
                    samples = param.sample_from_distribution(1, transformed=transformed)
                    if samples is not None:
                        param.update_initial_value(samples[0])
                        value = samples[0] if transformed else param.initial_value

                if value is None:
                    raise ParameterError(f"Parameter '{name}' has no initial value")

            if transformed:
                value = param.transformation.to_search(value)[0]

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
        if bounds is None or not np.isfinite(list(bounds.values())).all():
            raise ValueError("All parameters require bounds for plot.")

        return np.asarray(list(bounds.values())).T

    def to_dict(self, values: str | ArrayLike | None = None) -> Inputs:
        """
        Convert to parameter dictionary.

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
            return {name: param.initial_value for name, param in params}
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
                input_value = inputs[name]
                if input_value < param.bounds[0] or input_value > param.bounds[1]:
                    valid = False
        return valid

    def __repr__(self) -> str:
        param_summary = "\n".join(
            f" {name}: {param}" for name, param in self._parameters.items()
        )
        return f"Parameters({len(self)}):\n{param_summary}"

    def to_inputs(self, values: np.ndarray | list[np.ndarray]) -> list[Inputs]:
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
