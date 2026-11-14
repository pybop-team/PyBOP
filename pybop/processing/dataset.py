import warnings
from collections.abc import Iterator
from typing import Protocol

import numpy as np
import polars as pl
from polars._typing import FrameInitTypes
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
    data : dict or dataframe
        The experimental data to store within the dataset.
    domain : str, optional
        The domain of the dataset. Defaults to "Time [s]".
    control_functions : list[str], optional
        A list of function names for the control variables. Defaults to ["Current function [A]"].
    """

    def __init__(
        self,
        data: dict | FrameInitTypes,
        domain: str | None = None,
        control_functions: list[str] | None = None,
    ):
        """Initialise a Dataset instance with data and a set of names."""
        self.lf = data if isinstance(data, pl.LazyFrame) else pl.LazyFrame(data)
        if "column_0" in self.columns:  # missing column name(s)
            raise TypeError("pybop.Dataset expects a dictionary or dataframe.")
        self.domain = domain or "Time [s]"
        self.control_functions = control_functions or ["Current function [A]"]

    def __repr__(self):
        """Return a string representation of the Dataset instance."""
        return f"Dataset: {type(self.lf)} \n Contains: {self.columns}"

    def __setitem__(self, key, value):
        """Set the data corresponding to a particular key."""
        if key in self.columns:
            self.lf = self.lf.update(pl.LazyFrame({key: value}))
        else:
            self.lf = pl.concat((self.lf, pl.LazyFrame({key: value})), how="horizontal")

    def __getitem__(self, key):
        """Return the data corresponding to a particular key."""
        if key not in self.columns:
            if key == "Time [h]":
                return self["Time [s]"] / 3600
            raise ValueError(f"The key {key} does not exist in this dataset.")

        return self.lf.select(key).collect().to_numpy().ravel()

    @property
    def columns(self) -> list[str]:
        """Return the variable names as a list."""
        return self.lf.collect_schema().names()

    def keys(self) -> Iterator[str]:
        """Return the variable names."""
        return iter(self.columns)

    def __len__(self) -> int:
        """Return the length of the data, based on the length of the domain data."""
        return self.lf.select(pl.len()).collect().item()

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
        missing_attributes = set([self.domain, *signals]) - set(self.keys())
        if missing_attributes:
            raise ValueError(
                f"Expected {', '.join(missing_attributes)} in list of dataset"
            )

        domain_data = self[self.domain]

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
            if len(self[s]) != n_domain_data:
                raise ValueError(
                    f"{self.domain} data and {s} data must be the same length."
                )

    def get_subset(self, index: list | np.ndarray):
        """Reduce the dataset to a subset defined by the list of indices."""
        data = {}
        for key in self.keys():
            data[key] = self[key][index]

        return Dataset(
            data, domain=self.domain, control_functions=self.control_functions
        )

    def get_discontinuities(self, tol: float = 1e-6) -> np.ndarray:
        """Returns the start, end and any discontinuous points in the domain data."""
        discontinuities = [np.atleast_2d([0, len(self) - 1])]

        # Check all control variables for discontinuities
        for key in self.control_functions:
            control_data = self[key.replace(" function", "")]
            var_tol = np.max(np.abs(control_data)) * tol
            d = np.asarray(np.where(np.abs(np.diff(control_data)) > var_tol))
            discontinuities.append(d)
            discontinuities.append(d + 1)

        ind = np.unique(np.hstack(discontinuities))
        return self[self.domain][ind]

    def get_interpolant(self, control: str = "Current [A]") -> Interpolant:
        """Returns a linear interpolant for the control as a function of the domain."""

        # Check if a linearly interpolated current will match the charge throughput
        if control == "Current [A]" and "Discharge capacity [A.h]" in self.columns:
            charge_throughput = cumulative_trapezoid(
                y=self[control], x=self["Time [s]"]
            )
            if not np.allclose(
                self["Discharge capacity [A.h]"][1:] * 3600, charge_throughput
            ):
                warnings.warn(
                    "A linear interpolation of the current data does not reproduce the discharge capacity.",
                    stacklevel=2,
                )

        return Interpolant(self["Time [s]"], self[control], pybamm_t)


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


class Datasets:
    """
    Represents a collection of experimental observations.
    """

    def __init__(
        self,
        datasets: list[Dataset],
        domain: str | None = None,
        control_functions: list[str] | None = None,
    ):
        self.datasets = []
        self.domain = domain or "Time [s]"
        self.control_functions = control_functions or ["Current function [A]"]
        for dataset in datasets:
            if not isinstance(dataset, Dataset):
                dataset = Dataset(data_dictionary=dataset, domain=domain)
            self.datasets.append(dataset)

    def get_subset(self, indices):
        return Datasets(
            [self.datasets[i] for i in indices], self.domain, self.control_functions
        )

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count >= len(self):
            raise StopIteration
        return self.datasets[self.count]

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, i):
        return self.datasets[i]


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
