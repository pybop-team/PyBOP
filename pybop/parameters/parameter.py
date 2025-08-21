from __future__ import annotations

from collections.abc import Iterator, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pybop import (
    BasePrior,
    ComposedTransformation,
    IdentityTransformation,
    LogTransformation,
    Transformation,
)

# Type aliases
NumericValue = float | int | np.number
ArrayLike = Sequence[NumericValue] | NDArray[np.floating]
ParameterValue = NumericValue | ArrayLike | None
BoundsPair = list[float]
ParameterDict = dict[str, Any]
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


class ParameterValueValidator:
    """Validates and converts parameter values."""

    @staticmethod
    def validate_and_convert(value: Any, param_name: str) -> ParameterValue:
        """
        Validate and convert input value to appropriate type.

        Parameters
        ----------
        value : Any
            Input value to validate and convert
        param_name : str
            Parameter name for error messages

        Returns
        -------
        ParameterValue
            Validated and converted value

        Raises
        ------
        ParameterValidationError
            If value is invalid
        """
        if value is None:
            return None

        if isinstance(value, int | float | np.number):
            return float(value)

        if isinstance(value, list | tuple):
            if not value:
                raise ParameterValidationError(
                    f"Empty sequence not allowed for parameter '{param_name}'"
                )
            if not all(isinstance(x, int | float | np.number) for x in value):
                raise ParameterValidationError(
                    f"All elements must be numeric for parameter '{param_name}'"
                )
            return np.asarray(value, dtype=float)

        if isinstance(value, np.ndarray):
            if value.size == 0:
                raise ParameterValidationError(
                    f"Empty array not allowed for parameter '{param_name}'"
                )
            if not np.issubdtype(value.dtype, np.number):
                raise ParameterValidationError(
                    f"Array must contain numeric values for parameter '{param_name}'"
                )
            return value.astype(float)

        raise ParameterValidationError(
            f"Parameter value must be numeric, array-like, or None for parameter '{param_name}'. "
            f"Got {type(value)}"
        )

    @staticmethod
    def is_scalar(value: ParameterValue) -> bool:
        """Check if parameter value is scalar."""
        return value is None or not isinstance(value, np.ndarray)


class Parameter:
    """
    Represents a parameter within the PyBOP framework.

    This class encapsulates the definition of a parameter, including its name, prior
    distribution, initial value, bounds, and a margin to ensure the parameter stays
    within feasible limits during optimisation or sampling.

    """

    def __init__(
        self,
        name: str,
        *,
        initial_value: ParameterValue = None,
        current_value: ParameterValue = None,
        true_value: ParameterValue = None,
        bounds: BoundsPair | None = None,
        prior: BasePrior | None = None,
        transformation: Transformation | None = None,
        margin: float = 1e-4,
    ) -> None:
        """
        Initialise a Parameter.

        Parameters
        ----------
        name : str
            Parameter name
        initial_value : ParameterValue, optional
            Initial parameter value
        current_value : ParameterValue, optional
            Current parameter value (defaults to initial_value)
        true_value : ParameterValue, optional
            True parameter value (for testing/validation)
        bounds : tuple[float, float], optional
            Parameter bounds as (lower, upper)
        prior : Any, optional
            Prior distribution object
        transformation : Transformation, optional
            Parameter transformation
        margin : float, default=1e-4
            Safety margin for bounds sampling
        """
        self._name = str(name)
        self._prior = prior
        self._transformation = transformation or IdentityTransformation()
        self._validator = ParameterValueValidator()

        # Set bounds with validation
        self._bounds: Bounds | None = None
        if bounds is not None:
            self._bounds = Bounds(bounds[0], bounds[1])
        self._set_margin(margin)

        # Validate and set values
        if initial_value is None and self._prior is not None:
            initial_value = self.sample_from_prior()[0]
        self._initial_value = self._validator.validate_and_convert(initial_value, name)
        self._current_value = self._validator.validate_and_convert(
            current_value or initial_value, name
        )
        self._true_value = self._validator.validate_and_convert(true_value, name)

        # Validate initial values are within bounds
        self._validate_values_within_bounds()

    @property
    def name(self) -> str:
        return self._name

    @property
    def initial_value(self) -> ParameterValue:
        return self._copy_value(self._initial_value)

    @property
    def current_value(self) -> ParameterValue:
        return self._copy_value(self._current_value)

    @property
    def true_value(self) -> ParameterValue:
        return self._copy_value(self._true_value)

    @property
    def bounds(self) -> BoundsPair | None:
        """Parameter bounds as (lower, upper) tuple."""
        return (
            None if self._bounds is None else [self._bounds.lower, self._bounds.upper]
        )

    @property
    def prior(self) -> Any | None:
        return self._prior

    @property
    def transformation(self) -> Transformation:
        return self._transformation

    @property
    def is_scalar(self) -> bool:
        """Whether the parameter value is scalar."""
        return self._validator.is_scalar(self._current_value)

    def _copy_value(self, value: ParameterValue) -> ParameterValue:
        """Create a copy of the parameter value."""
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return value.copy()
        return value

    def _set_margin(self, margin: float) -> None:
        """Set sampling margin with validation."""
        if not 0 < margin < 1:
            raise ParameterValidationError("Margin must be between 0 and 1")
        self._margin = margin

    def _validate_values_within_bounds(self) -> None:
        """Validate that initial values are within bounds."""
        if self._bounds is None or self._initial_value is None:
            return

        if self.is_scalar:
            if not self._bounds.contains(self._initial_value):
                raise ParameterValidationError(
                    f"Parameter '{self._name}': Initial value {self._initial_value} "
                    f"is outside bounds {self.bounds}"
                )
        else:
            if not self._bounds.contains_array(self._initial_value):
                raise ParameterValidationError(
                    f"Parameter '{self._name}': Some initial values are outside bounds {self.bounds}"
                )

    def update_value(self, value: ParameterValue) -> None:
        """
        Update the current parameter value.

        Parameters
        ----------
        value : ParameterValue
            New parameter value
        """
        validated_value = self._validator.validate_and_convert(value, self._name)
        self._current_value = validated_value

    def update_initial_value(self, value: ParameterValue) -> None:
        """
        Update the initial parameter value.

        Parameters
        ----------
        value : ParameterValue
            New initial value
        """
        validated_value = self._validator.validate_and_convert(value, self._name)
        self._initial_value = validated_value
        # Also update current value if it was None
        if self._current_value is None:
            self._current_value = self._copy_value(validated_value)

    def set_bounds(self, bounds: BoundsPair) -> None:
        """
        Set new parameter bounds.

        Parameters
        ----------
        bounds : tuple[float, float]
            New bounds as (lower, upper)
        """
        self._bounds = Bounds(bounds[0], bounds[1])

    def reset_to_initial(self) -> None:
        """Reset current value to initial value."""
        if self._initial_value is None:
            raise ParameterError(
                f"Parameter '{self._name}' has no initial value to reset to"
            )
        self._current_value = self._copy_value(self._initial_value)

    def sample_from_prior(
        self,
        n_samples: int = 1,
        *,
        random_state: int | None = None,
        apply_transform: bool = False,
    ) -> NDArray[np.floating] | None:
        """
        Sample from parameter's prior distribution.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to draw
        random_state : int, optional
            Random seed for reproducibility
        apply_transform : bool, default=False
            Whether to apply transformation to samples

        Returns
        -------
        NDArray[np.floating] or None
            Array of samples, or None if no prior exists
        """
        if self._prior is None:
            return None

        samples = self._prior.rvs(n_samples, random_state=random_state)
        samples = np.atleast_1d(samples).astype(float)

        # Apply bounds clipping if bounds exist
        if self._bounds is not None:
            offset = self._margin * self._bounds.width()
            effective_lower = self._bounds.lower + offset
            effective_upper = self._bounds.upper - offset
            samples = np.clip(samples, effective_lower, effective_upper)

        if apply_transform:
            samples = np.array([self._transformation.to_search(s)[0] for s in samples])

        return samples

    def get_initial_value_transformed(self) -> NDArray | None:
        """Get initial value in transformed space."""
        if self._initial_value is None:
            return None
        if not self.is_scalar:
            raise ParameterError("Transformation only supported for scalar parameters")
        return self._transformation.to_search(self._initial_value)[0]

    def __repr__(self) -> str:
        """String representation of the parameter."""
        return f"Parameter: {self.name} \n Prior: {self.prior} \n Bounds: {self.bounds} \n Value: {self.current_value}"

    def __eq__(self, other: object) -> bool:
        """Check equality based on name."""
        if not isinstance(other, Parameter):
            return False
        return self._name == other._name

    def __hash__(self) -> int:
        """Hash based on name."""
        return hash(self._name)


class Parameters:
    """
    Container for managing multiple Parameter objects with additional functionality.

    This class provides a comprehensive interface for parameter management including
    validation, transformation, serialisation, and bulk operations.
    """

    def __init__(self, parameters: Sequence[Parameter] | None = None) -> None:
        """
        Initialise Parameters container.

        Parameters
        ----------
        parameters : Sequence[Parameter], optional
            Initial parameters to add
        """
        self._parameters: dict[str, Parameter] = {}

        if parameters:
            for param in parameters:
                self._add(param, update_transform=False)

        self._transform = self._construct_transformation()

    def _add(self, parameter: Parameter, update_transform: bool = True) -> None:
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

        if parameter.name in self._parameters:
            raise ParameterError(f"Parameter '{parameter.name}' already exists")

        self._parameters[parameter.name] = parameter

        if update_transform:
            self._transform = self._construct_transformation()

    def add(self, parameter: Parameter) -> None:
        """Add a parameter to the collection."""
        self._add(parameter)

    def remove(self, name: str) -> Parameter:
        """Remove parameter and return it."""
        if name not in self._parameters:
            raise ParameterNotFoundError(f"Parameter '{name}' not found")
        return self._parameters.pop(name)

    def get(self, name: str) -> Parameter:
        """Get a parameter by name."""
        if name not in self._parameters:
            raise ParameterNotFoundError(f"Parameter '{name}' not found")
        return self._parameters[name]

    def update(
        self,
        *,
        values: ArrayLike | ParameterDict | None = None,
        initial_values: ArrayLike | ParameterDict | None = None,
        bounds: Sequence[BoundsPair] | dict[str, BoundsPair] | None = None,
        **individual_updates: dict[str, Any],
    ) -> None:
        """
        Update multiple parameters efficiently.

        Parameters
        ----------
        values : array-like or dict, optional
            New current values (by position or name)
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
                if "value" in updates:
                    param.update_value(updates["value"])
                if "initial_value" in updates:
                    param.update_initial_value(updates["initial_value"])
                if "bounds" in updates:
                    param.set_bounds(updates["bounds"])
            else:
                param.update_value(updates)

        # Handle bulk updates
        if values is not None:
            self._bulk_update_values(values)
        if initial_values is not None:
            self._bulk_update_initial_values(initial_values)
        if bounds is not None:
            self._bulk_update_bounds(bounds)

    def _bulk_update_values(self, values: ArrayLike | ParameterDict) -> None:
        """Update current values in bulk."""
        if isinstance(values, dict):
            for name, value in values.items():
                self.get(name).update_value(value)
        else:
            values_array = np.atleast_1d(values)
            param_list = list(self._parameters.values())

            for param, value in zip(param_list, values_array.T, strict=False):
                param.update_value(value)

    def _bulk_update_initial_values(self, values: ArrayLike | ParameterDict) -> None:
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

    def _bulk_update_bounds(
        self, bounds: Sequence[BoundsPair] | dict[str, BoundsPair]
    ) -> None:
        """Update bounds in bulk."""
        if isinstance(bounds, dict):
            for name, bound_pair in bounds.items():
                self.get(name).set_bounds(bound_pair)
        else:
            param_list = list(self._parameters.values())

            if len(bounds) != len(param_list):
                raise ParameterValidationError(
                    f"Bounds array length {len(bounds)} doesn't match "
                    f"parameter count {len(param_list)}"
                )

            for param, bound_pair in zip(param_list, bounds, strict=False):
                param.set_bounds(bound_pair)

    def get_values(self, *, transformed: bool = False) -> NDArray[np.floating]:
        """
        Get current values as array.

        Parameters
        ----------
        transformed : bool, default=False
            Whether to apply transformations

        Returns
        -------
        NDArray[np.floating]
            Array of current values
        """
        values = []
        for param in self._parameters.values():
            value = param.current_value
            if value is None:
                raise ParameterError(f"Parameter '{param.name}' has no current value")

            if transformed:
                if not param.is_scalar:
                    value = [param.transformation.to_search(val)[0] for val in value]
                else:
                    value = param.transformation.to_search(value)[0]

            values.append(value)

        return np.asarray(values)

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
        for param in self._parameters.values():
            value = param.initial_value
            if value is None:
                # Try to sample from prior if available
                if param.prior is not None:
                    samples = param.sample_from_prior(1, apply_transform=transformed)
                    if samples is not None:
                        param.update_initial_value(samples[0])
                        value = samples[0] if transformed else param.initial_value

                if value is None:
                    raise ParameterError(
                        f"Parameter '{param.name}' has no initial value"
                    )

            if transformed and param.is_scalar:
                value = param.transformation.to_search(value)[0]

            values.append(value)

        return np.asarray(values)

    def get_bounds(self, *, transformed: bool = False) -> dict[str, list[float]]:
        """
        Get parameter bounds.

        Parameters
        ----------
        transformed : bool, default=False
            Whether to apply transformations to bounds

        Returns
        -------
        dict[str, list[float]]
            Dictionary with 'lower' and 'upper' keys containing bound arrays
        """
        lower_bounds = []
        upper_bounds = []

        for param in self._parameters.values():
            if param.bounds is None:
                lower, upper = -np.inf, np.inf
            else:
                lower, upper = param.bounds

                if transformed and param.is_scalar:
                    # Handle special case for log transformation at zero
                    if (
                        isinstance(param.transformation, LogTransformation)
                        and lower == 0
                    ):
                        lower_t = -np.inf
                    else:
                        lower_t = param.transformation.to_search(lower)[0]

                    upper_t = param.transformation.to_search(upper)[0]
                    lower, upper = min(lower_t, upper_t), max(lower_t, upper_t)

            lower_bounds.append(lower)
            upper_bounds.append(upper)

        return {"lower": lower_bounds, "upper": upper_bounds}

    def get_bounds_array(self, apply_transform: bool = False) -> np.ndarray:
        """
        Retrieve parameter bounds in numpy format.

        Returns
        -------
        bounds : numpy.ndarray
            An array of shape (n_parameters, 2) containing the bounds for each parameter.
        """
        bounds = self.get_bounds(transformed=apply_transform)
        return np.column_stack([bounds["lower"], bounds["upper"]])

    def _construct_transformation(self) -> Transformation:
        """
        Create a ComposedTransformation object from the individual parameter transformations.
        """
        transformations = []

        for param in self._parameters.values():
            transformations.append(param.transformation)

        return ComposedTransformation(transformations)

    def sample_from_priors(
        self,
        n_samples: int = 1,
        *,
        random_state: int | None = None,
        transformed: bool = False,
    ) -> NDArray[np.floating] | None:
        """
        Sample from all parameter priors.

        Returns
        -------
        NDArray[np.floating] or None
            Array of shape (n_samples, n_parameters) or None if any prior is missing
        """
        all_samples = []

        for param in self._parameters.values():
            samples = param.sample_from_prior(
                n_samples, random_state=random_state, apply_transform=transformed
            )
            if samples is None:
                return None
            all_samples.append(samples)

        return np.column_stack(all_samples)

    def to_dict(self, values: str | ArrayLike | None = None) -> ParameterDict:
        """
        Convert to parameter dictionary.

        Parameters
        ----------
        values : str or array-like, optional
            Which values to use ('current', 'initial', 'true') or custom array

        Returns
        -------
        ParameterDict
            Dictionary mapping parameter names to values
        """
        params = self._parameters.items()
        if values is None:
            values = "current"
        if isinstance(values, str):
            if values == "current":
                return {name: param.current_value for name, param in params}
            elif values == "initial":
                return {name: param.initial_value for name, param in params}
            elif values == "true":
                return {name: param.true_value for name, param in params}
        else:
            # Custom values array
            values_array = np.atleast_1d(values)
            if len(values_array) != len(self._parameters):
                raise ParameterValidationError(
                    "Values array length doesn't match parameter count"
                )
            return dict(zip(self._parameters.keys(), values_array, strict=False))

    def to_pybamm_multiprocessing(self) -> list:
        """
        Return parameter values as a list of dictionaries in the format
        required for pybamm multiprocessing.
        """
        param_dict = self.to_dict()

        if self.get_values().ndim == 1:
            return [param_dict]

        # Construct a list of single value dicts
        array_length = len(next(iter(param_dict.values())))
        return [
            {key: float(values[i]) for key, values in param_dict.items()}
            for i in range(array_length)
        ]

    def reset_to_initial(self, names: Sequence[str] | None = None) -> None:
        """Reset parameters to initial values."""
        target_params = (
            [self.get(name) for name in names]
            if names
            else list(self._parameters.values())
        )

        for param in target_params:
            param.reset_to_initial()

    def priors(self) -> list:
        """Return the prior distribution of each parameter."""
        return [
            param.prior
            for param in self._parameters.values()
            if param.prior is not None
        ]

    @property
    def transformation(self) -> Transformation:
        """
        Get the transformation for the parameters.
        """
        return self._transform

    def copy(self) -> Parameters:
        """Create a deep copy of the Parameters object."""
        return deepcopy(self)

    def __len__(self) -> int:
        return len(self._parameters)

    def __contains__(self, name: str) -> bool:
        return name in self._parameters

    def __getitem__(self, name: str) -> Parameter:
        return self.get(name)

    def __iter__(self) -> Iterator[Parameter]:
        return iter(self._parameters.values())

    def keys(self) -> Iterator[str]:
        """Iterate over parameter names."""
        return iter(self._parameters.keys())

    def values(self) -> Iterator[Parameter]:
        """Iterate over parameters."""
        return iter(self._parameters.values())

    def items(self) -> Iterator[tuple[str, Parameter]]:
        """Iterate over (name, parameter) pairs."""
        return iter(self._parameters.items())

    def __repr__(self) -> str:
        param_summary = "\n".join(
            f" {name}: prior= {param.prior}, value={param.current_value}, bounds={param.bounds}"
            for name, param in self._parameters.items()
        )
        return f"Parameters({len(self)}):\n{param_summary}"
