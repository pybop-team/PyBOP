import re
import warnings
from typing import Protocol

import numpy as np
from pybamm import Interpolant, Solution
from pybamm import t as pybamm_t
from scipy.integrate import cumulative_trapezoid


class PyprobeResult(Protocol):
    """Protocol defining required PyProBE Result interface."""

    def get(
        self,
        *column_names: str,
    ) -> np.typing.NDArray[np.float64] | tuple[np.typing.NDArray[np.float64], ...]:
        """Get result data as numpy ndarray"""

    @property
    def columns(self) -> list[str]:
        """List of column data"""


class Dataset:
    """
    Represents a collection of experimental observations.

    This class provides a structured way to store and work with experimental data,
    which may include applying operations such as interpolation.

    Parameters
    ----------
    data_dictionary : dict
        The experimental data to store within the dataset.
    domain : str, optional
        The domain of the dataset. Defaults to "Time [s]".
    control_functions : list[str], optional
        A list of function names for the control variables. Defaults to ["Current function [A]"].
    """

    def __init__(
        self,
        data_dictionary: dict,
        domain: str | None = None,
        control_functions: list[str] | None = None,
    ):
        """Initialise a Dataset instance with data and a set of names."""
        if not isinstance(data_dictionary, dict):
            raise TypeError("The input to pybop.Dataset must be a dictionary.")
        self.data = data_dictionary
        self.domain = domain or "Time [s]"
        self.control_functions = control_functions or ["Current function [A]"]

    def __repr__(self):
        """Return a string representation of the Dataset instance."""
        return f"Dataset: {type(self.data)} \n Contains: {self.data.keys()}"

    def __setitem__(self, key, value):
        """Set the data corresponding to a particular key."""
        self.data[key] = value

    def __getitem__(self, key):
        """Return the data corresponding to a particular key."""
        if key not in self.data.keys():
            raise ValueError(f"The key {key} does not exist in this dataset.")

        return self.data[key]

    def keys(self):
        """
        Return the keys of the data dictionary.

        Returns
        -------
        dict_keys
        """
        return self.data.keys()

    def __len__(self) -> int:
        """Return the length of the data, based on the length of the domain data."""
        return len(self.data[self.domain])

    def check(self, domain: str = None, signal: str | list[str] = None) -> bool:
        """
        Check the consistency of a PyBOP Dataset against the expected format.

        Parameters
        ----------
        domain : str, optional
            If not None, updates the domain of the dataset.
        signal : str or List[str], optional
            The signal(s) to check. Defaults to ["Voltage [V]"].

        Returns
        -------
        bool
            True if the dataset has the expected attributes.

        Raises
        ------
        ValueError
            If the time series and the data series are not consistent.
        """
        self.domain = domain or self.domain
        signals = [signal] if isinstance(signal, str) else (signal or ["Voltage [V]"])

        # Check that the dataset contains domain and chosen signals
        missing_attributes = set([self.domain, *signals]) - set(self.data.keys())
        if missing_attributes:
            raise ValueError(
                f"Expected {', '.join(missing_attributes)} in list of dataset"
            )

        domain_data = self.data[self.domain]

        # Check domain-specific constraints
        if self.domain == "Time [s]":
            self._check_time_constraints(domain_data)
        elif self.domain == "Frequency [Hz]":
            self._check_frequency_constraints(domain_data)

        # Check for consistent data length
        self._check_data_consistency(domain_data, signals)

        return True

    @staticmethod
    def _check_time_constraints(time_data: np.ndarray) -> None:
        if np.any(time_data < 0):
            raise ValueError("Times cannot be negative.")
        if np.any(time_data[:-1] >= time_data[1:]):
            raise ValueError("Times must be increasing.")

    @staticmethod
    def _check_frequency_constraints(freq_data: np.ndarray) -> None:
        if np.any(freq_data < 0):
            raise ValueError("Frequencies cannot be negative.")

    def _check_data_consistency(
        self, domain_data: np.ndarray, signals: list[str]
    ) -> None:
        n_domain_data = len(domain_data)
        for s in signals:
            if np.shape(self.data[s])[-1] != n_domain_data:
                raise ValueError(
                    f"{self.domain} data and {s} data must be the same length."
                )

    def get_subset(self, index: list | np.ndarray):
        """Reduce the dataset to a subset defined by the list of indices."""
        data = {}
        for key in self.data.keys():
            data[key] = self[key][index]

        return Dataset(
            data, domain=self.domain, control_functions=self.control_functions
        )

    def get_discontinuities(self, tol: float = 1e-6) -> np.ndarray:
        """Returns the start, end and any discontinuous points in the domain data."""
        discontinuities = [np.atleast_2d([0, len(self.data[self.domain]) - 1])]

        # Check all control variables for discontinuities
        for key in self.control_functions:
            control_data = self.data[key.replace(" function", "")]
            var_tol = np.max(np.abs(control_data)) * tol
            d = np.asarray(np.where(np.abs(np.diff(control_data)) > var_tol))
            discontinuities.append(d)
            discontinuities.append(d + 1)

        ind = np.unique(np.hstack(discontinuities))
        return self.data[self.domain][ind]

    def get_interpolant(self, control: str = "Current [A]") -> Interpolant:
        """Returns a linear interpolant for the control as a function of the domain."""

        # Check if a linearly interpolated current will match the charge throughput
        if control == "Current [A]" and "Discharge capacity [A.h]" in self.data.keys():
            charge_throughput = cumulative_trapezoid(
                y=self.data[control], x=self.data["Time [s]"]
            )
            if not np.allclose(
                self.data["Discharge capacity [A.h]"][1:] * 3600, charge_throughput
            ):
                warnings.warn(
                    "A linear interpolation of the current data does not reproduce the discharge capacity.",
                    stacklevel=2,
                )

        return Interpolant(self.data["Time [s]"], self.data[control], pybamm_t)


def get_impedance_variables(frequencies: np.ndarray | list[float]) -> list[str]:
    """
    Return the names of the dataset variables which hold an impedance spectrum, being
    two real-valued variables, the real and the imaginary component, per frequency.

    Complex data is split into real components because the error measures square the
    residual, and `r**2` is not `abs(r)**2` for a complex array.

    Parameters
    ----------
    frequencies : np.ndarray or list[float]
        The frequencies at which the impedance is evaluated.

    Returns
    -------
    list[str]
        Variable names, ordered (real, imaginary) for each frequency in turn.
    """
    return [
        f"Impedance {component} [Ohm] ({frequency:.6g} Hz)"
        for frequency in frequencies
        for component in ("real", "imaginary")
    ]


def parse_impedance_variables(
    variables: list[str],
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Pick out the impedance variables from a list of variable names, inverting
    :func:`get_impedance_variables`.

    Parameters
    ----------
    variables : list[str]
        Variable names, which may or may not include impedance variables.

    Returns
    -------
    tuple[np.ndarray, list[str], list[str]]
        The frequencies in increasing order, and the corresponding real and imaginary
        variable names. All three are empty if no impedance variables are present.
    """
    pattern = re.compile(r"^Impedance (real|imaginary) \[Ohm\] \((\S+) Hz\)$")

    frequencies = {}
    for variable in variables:
        match = pattern.match(variable)
        if match:
            component, frequency = match.group(1), float(match.group(2))
            frequencies.setdefault(frequency, {})[component] = variable

    # Ignore any frequency which is missing one of its two components
    ordered = sorted(f for f, components in frequencies.items() if len(components) == 2)
    return (
        np.asarray(ordered),
        [frequencies[f]["real"] for f in ordered],
        [frequencies[f]["imaginary"] for f in ordered],
    )


def import_pybamm_solution(
    solution: Solution,
    variables: list[str] | None = None,
    t_interp: np.ndarray | None = None,
    domain: str | None = None,
    control_functions: list[str] | None = None,
) -> Dataset:
    """
    Import a pybamm.Solution into a pybop.Dataset.

    Parameters
    ----------
    solution : pybamm.Solution
        A pybamm.Solution object.
    variables : list[str], optional
        A list of variables to include in the dataset.
    t_interp : np.ndarray, optional
        Time points at which to interpolate the solution, only when the solver supports it.
    domain : str, optional
        The domain of the dataset. Defaults to "Time [s]".
    control_functions : list[str], optional
        A list of function names for the control variables. Defaults to ["Current function [A]"].
    """
    if variables is None:
        variables = [
            "Time [s]",
            "Current [A]",
            "Voltage [V]",
            "Discharge capacity [A.h]",
        ]

    if t_interp is None:
        data_dict = solution.get_data_dict(variables=variables)
    else:
        data_dict = {}
        for key in variables:
            data_dict[key] = solution[key](t_interp)

    return Dataset(data_dict, domain=domain, control_functions=control_functions)


def import_pyprobe_result(
    result: PyprobeResult,
    variables: list[str] | None = None,
    column_names: list[str] | None = None,
    domain: str | None = None,
    control_functions: list[str] | None = None,
) -> Dataset:
    """
    Import a pyprobe.Result into a pybop.Dataset.

    Parameters
    ----------
    result : PyprobeResult | pyprobe.Result
        A pyprobe.Result-like object.
    variables : list[str], optional
        A list of variables to include in the dataset.
    column_names : list[str], optional
        A list of the column names in the Result corresponding to the variable names.
        If only one list of names is provided, they are assumed to be identical.
    domain : str, optional
        The domain of the dataset. Defaults to "Time [s]".
    control_functions : list[str], optional
        A list of function names for the control variables. Defaults to ["Current function [A]"].
    """
    if variables is None and column_names is None:
        variables = [
            "Time [s]",
            "Current [A]",
            "Voltage [V]",
            "Discharge capacity [A.h]",
        ]
        column_names = [
            "Time [s]",
            "Current [A]",
            "Voltage [V]",
            "Capacity [Ah]",
        ]
    elif variables is None:
        variables = column_names
    elif column_names is None:
        column_names = variables

    data_dict = {}
    for i, col in enumerate(variables):
        if (
            column_names[i] == "Cycle"
            and "Cycle" not in result.columns
            and "Step" in result.columns
        ):
            warnings.warn(
                "No cycle information present. Cycles will be inferred from the step numbers.",
                UserWarning,
                stacklevel=2,
            )
            steps = result.get("Step")
            cycle_ends = np.argwhere(steps - np.roll(steps, 1) < 0)
            cycle_ends = np.append(cycle_ends, len(steps))
            data_dict[col] = np.concatenate(
                [
                    (i - 1) * np.ones(cycle_ends[i] - cycle_ends[i - 1])
                    for i in range(1, len(cycle_ends))
                ]
            )
        elif column_names[i] in ["Current [A]", "Capacity [Ah]"]:
            # The sign convention in PyProBE is that positive current is charging,
            # the convention in PyBaMM is that positive current means discharging
            data_dict[col] = -1.0 * result.get(column_names[i])
        else:
            data_dict[col] = result.get(column_names[i])
    return Dataset(data_dict, domain=domain, control_functions=control_functions)
