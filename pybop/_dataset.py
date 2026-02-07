import warnings
from typing import Protocol

import numpy as np
from pybamm import Interpolant, Solution
from pybamm import t as pybamm_t


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
            if len(self.data[s]) != n_domain_data:
                raise ValueError(
                    f"{self.domain} data and {s} data must be the same length."
                )

    def get_subset(self, index: list | np.ndarray):
        """Reduce the dataset to a subset defined by the list of indices."""
        data = {}
        for key in self.data.keys():
            data[key] = self[key][index]

        return Dataset(data, domain=self.domain)

    def get_interpolant(self, control: str = "Current [A]") -> Interpolant:
        """Returns a linear interpolant for the control as a function of the domain."""
        return Interpolant(self.data["Time [s]"], self.data[control], pybamm_t)


def import_pybamm_solution(
    solution: Solution,
    required_columns: list[str] | None = None,
    original_columns: list[str] | None = None,
) -> Dataset:
    """
    Import a pybamm.Solution into a pybop.Dataset.

    Parameters
    ----------
    solution : pybamm.Solution
        A pybamm.Solution object.
    required_columns : list[str], optional
        List of column names for the pybop.Dataset.
    original_columns : list[str], optional
        A list of the column names in the Result corresponding to the required column names.
    If only one list of column names is provided, they are assumed to be identical.
    """
    if required_columns is None and original_columns is None:
        required_columns = [
            "Time [s]",
            "Current [A]",
            "Voltage [V]",
            "Discharge capacity [A.h]",
        ]
        original_columns = [
            "Time [s]",
            "Current [A]",
            "Voltage [V]",
            "Discharge capacity [A.h]",
        ]
    elif required_columns is None:
        required_columns = original_columns
    elif original_columns is None:
        original_columns = required_columns

    data_dict = solution.get_data_dict(variables=original_columns)

    for old_key, new_key in zip(original_columns, required_columns, strict=False):
        data_dict[new_key] = data_dict.pop(old_key)
    return Dataset(data_dict)


def import_pyprobe_result(
    result: PyprobeResult,
    required_columns: list[str] | None = None,
    original_columns: list[str] | None = None,
) -> Dataset:
    """
    Import a pyprobe.Result into a pybop.Dataset.

    Parameters
    ----------
    result : PyprobeResult | pyprobe.Result
        A pyprobe.Result-like object.
    required_columns : list[str], optional
        List of column names for the pybop.Dataset.
    original_columns : list[str], optional
        A list of the column names in the Result corresponding to the required column names.
    If only one list of column names is provided, they are assumed to be identical.
    """
    if required_columns is None and original_columns is None:
        required_columns = [
            "Time [s]",
            "Current [A]",
            "Voltage [V]",
            "Discharge capacity [A.h]",
        ]
        original_columns = [
            "Time [s]",
            "Current [A]",
            "Voltage [V]",
            "Capacity [Ah]",
        ]
    elif required_columns is None:
        required_columns = original_columns
    elif original_columns is None:
        original_columns = required_columns

    data_dict = {}
    for i, col in enumerate(required_columns):
        if (
            original_columns[i] == "Cycle"
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
        elif original_columns[i] in ["Current [A]", "Capacity [Ah]"]:
            # The sign convention in PyProBE is that positive current is charging,
            # the convention in PyBaMM is that positive current means discharging
            data_dict[col] = -1.0 * result.get(original_columns[i])
        else:
            data_dict[col] = result.get(original_columns[i])
    return Dataset(data_dict)
