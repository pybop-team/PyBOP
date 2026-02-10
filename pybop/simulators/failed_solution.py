from dataclasses import dataclass, field

import numpy as np
import pybamm


@dataclass(frozen=True)
class FailedVariable:
    """
    Container for a failed PyBaMM variable that returns np.inf.

    Args:
        name: Variable name
        data: Array data, defaults to [np.inf]
        sensitivities: Sensitivity data mapping parameter names to arrays
    """

    name: str
    data: np.ndarray = field(default_factory=lambda: np.asarray([np.inf]))
    sensitivities: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate inputs after initialisation."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Variable name must be a non-empty string")

        if not isinstance(self.data, np.ndarray):
            object.__setattr__(self, "data", np.asarray(self.data))


class FailedSolution:
    """
    Container for a failed PyBaMM solution that returns [np.inf] for all processed variables.

    This class mimics the interface of a successful PyBaMM solution but returns
    infinity values to indicate failure while maintaining API compatibility.

    Args:
        variable_names: List of variable names in the solution
        parameter_names: List of parameter names for sensitivity analysis

    Example:
        >>> solution = FailedSolution(["Voltage [V]"], ["Negative particle radius [m]"])
        >>> voltage = solution["Voltage [V]"]
        >>> print(voltage.data)  # np.ndarray([inf])
    """

    def __init__(self, variable_names: list[str], parameter_names: list[str]):
        self._validate_inputs(variable_names, parameter_names)
        self._variable_names = variable_names
        self._parameter_names = parameter_names

        # Solution metadata
        self.cycles: int | None = None
        self.termination: str = "failure"
        self.solve_time: float = 0.0
        self.integration_time: float = 0.0
        self._t_eval: np.ndarray = np.asarray([0.0])

        # Initialise failed variables
        self._variables: dict[str, FailedVariable] = pybamm.FuzzyDict()
        self._initialise_variables()

    def _validate_inputs(
        self, variable_names: list[str], parameter_names: list[str] | None
    ) -> None:
        """Validate constructor inputs."""
        if not variable_names:
            raise ValueError("variable_names cannot be empty")

        if not all(isinstance(name, str) and name.strip() for name in variable_names):
            raise ValueError("All variable names must be non-empty strings")

        if parameter_names is not None:
            if not all(
                isinstance(name, str) and name.strip() for name in parameter_names
            ):
                raise ValueError("All parameter names must be non-empty strings")

    def _initialise_variables(self) -> None:
        """Initialise all variables with failed state."""
        inf_array = np.asarray([np.inf])

        for var_name in self._variable_names:
            if self._parameter_names:
                sensitivities = {p: inf_array.copy() for p in self._parameter_names}
                sensitivities["all"] = [inf_array.copy() for _ in self._parameter_names]
            else:
                sensitivities = {}

            self._variables[var_name] = FailedVariable(
                name=var_name, data=inf_array.copy(), sensitivities=sensitivities
            )

    def __getattr__(self, name):
        # Return self for any method calls to allow chaining
        return self

    def __getitem__(self, key):
        return self._variables[key]

    def plot(self, *args, **kwargs):
        print("Cannot plot a failed solution")
        return None

    def save(self, *args, **kwargs):
        print("Cannot save a failed solution")
        return None

    def copy(self):
        return FailedSolution(self._variable_names, self._parameter_names)

    @property
    def t_eval(self) -> np.ndarray:
        """Time evaluation points (returns [inf] for failed solutions)."""
        return self._t_eval

    @property
    def variable_names(self) -> list[str]:
        """Get list of variable names (read-only)."""
        return self._variable_names.copy()

    @property
    def parameter_names(self) -> list[str]:
        """Get list of parameter names (read-only)."""
        return self._parameter_names.copy()

    def keys(self) -> list[str]:
        """Get all variable names."""
        return list(self._variables.keys())

    def values(self) -> list[FailedVariable]:
        """Get all variables."""
        return list(self._variables.values())

    def items(self) -> list[tuple[str, FailedVariable]]:
        """Get all variable name-value pairs."""
        return list(self._variables.items())
