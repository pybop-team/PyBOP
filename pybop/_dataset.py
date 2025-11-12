import warnings
from typing import Protocol

import numpy as np
from pybamm import solvers


class PyprobeResult(Protocol):
    """Protocol defining required PyProBE Result interface"""

    def get(
        self,
        *column_names: str,
    ) -> np.typing.NDArray[np.float64] | tuple[np.typing.NDArray[np.float64], ...]:
        """Get result data as numpy ndarray"""

    @property
    def column_list(self) -> list[str]:
        """List of column data"""


class Dataset:
    """
    Represents a collection of experimental observations.

    This class provides a structured way to store and work with experimental data,
    which may include applying operations such as interpolation.

    Parameters
    ----------
    data_dictionary : dict or instance of pybamm.solvers.solution.Solution
        The experimental data to store within the dataset.
    domain : str, optional
        The domain of the dataset. Defaults to "Time [s]".
    """

    def __init__(
        self,
        data_dictionary,
        domain: str | None = None,
        variables: str | None = ["Time [s]", "Current [A]", "Voltage [V]"],
    ):
        """
        Initialise a Dataset instance with data and a set of names.
        """

        if isinstance(data_dictionary, solvers.solution.Solution):
            data_dictionary = data_dictionary.get_data_dict(variables=variables)
        if not isinstance(data_dictionary, dict):
            raise TypeError(
                "The input to pybop.Dataset must be a dictionary or a pybamm.Solution object."
            )
        self.data = data_dictionary
        self.domain = domain or "Time [s]"

    def __repr__(self):
        """
        Return a string representation of the Dataset instance.

        Returns
        -------
        str
            A string that includes the type and contents of the dataset.
        """
        return f"Dataset: {type(self.data)} \n Contains: {self.data.keys()}"

    def __setitem__(self, key, value):
        """
        Set the data corresponding to a particular key.

        Parameters
        ----------
        key : str
            The name of the key to be set.

        value : list or np.ndarray
            The data series to be stored in the dataset.
        """
        self.data[key] = value

    def __getitem__(self, key):
        """
        Return the data corresponding to a particular key.

        Parameters
        ----------
        key : str
            The name of a data series within the dataset.

        Returns
        -------
        list or np.ndarray
            The data series corresponding to the key.

        Raises
        ------
        ValueError
            The key must exist in the dataset.
        """
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
        """
        Reduce the dataset to a subset defined by the list of indices.
        """
        data = {}
        for key in self.data.keys():
            data[key] = self[key][index]

        return Dataset(data, domain=self.domain)


def import_pyprobe_result(
    result: PyprobeResult,
    pybop_columns: list[str] | None = None,
    pyprobe_columns: list[str] | None = None,
) -> Dataset:
    """
    Import a pyprobe.Result into a dictionary

    Parameters
    ----------
    result : str
        A pyprobe.Result object.
    pybop_columns : list[str]
        List of pybop column names.
    pyprobe_columns : list[str]
        An list of pyprobe column names.
    If only one list of column names is provided, they are assumed to be identical.
    """
    if pybop_columns is None and pyprobe_columns is None:
        pybop_columns = [
            "Time [s]",
            "Current function [A]",
            "Voltage [V]",
            "Discharge capacity [Ah]",
        ]
        pyprobe_columns = [
            "Time [s]",
            "Current [A]",
            "Voltage [V]",
            "Capacity [Ah]",
        ]
    elif pybop_columns is None:
        pybop_columns = pyprobe_columns
    elif pyprobe_columns is None:
        pyprobe_columns = pybop_columns

    data_dict = {}
    for i, col in enumerate(pybop_columns):
        if (
            pyprobe_columns[i] == "Cycle"
            and "Cycle" not in result.column_list
            and "Step" in result.column_list
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
        elif pyprobe_columns[i] in [
            "Current [A]",
            "Capacity [Ah]",
        ]:
            # The sign convention in PyProBE is that positive current is charging,
            # the convention in PyBaMM is that positive current means discharging
            data_dict[col] = -1.0 * result.get(pyprobe_columns[i])
        else:
            data_dict[col] = result.get(pyprobe_columns[i])
    return Dataset(data_dict)
