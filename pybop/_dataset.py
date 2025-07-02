import copy

import numpy as np
from pybamm import Solution


class Dataset:
    """
    Represents a collection of experimental observations.

    This class provides a structured way to store and work with experimental data,
    which may include applying operations such as interpolation.

    Parameters
    ----------
    data_dictionary : dict[str, np.ndarray|list] or instance of pybamm.Solution
        The experimental data to store within the dataset.
    domain : str, optional
        The domain of the dataset. Defaults to "Time [s]".
    """

    def __init__(
        self,
        data_dictionary: dict[str, np.ndarray | list] | Solution,
        domain: str | None = None,
        control_variable: str | None = None,
    ):
        """
        Initialise a Dataset instance with data and a set of names.
        """

        if isinstance(data_dictionary, Solution):
            data_dictionary = data_dictionary.get_data_dict()
        if not isinstance(data_dictionary, dict):
            raise TypeError("The input to pybop.Dataset must be a dictionary.")
        # convert any lists to numpy arrays
        data_ndarray: dict[str, np.ndarray] = {}
        for key, value in data_dictionary.items():
            if isinstance(value, list):
                data_ndarray[key] = np.array(value)
            elif isinstance(value, np.ndarray):
                data_ndarray[key] = value
            else:
                raise TypeError(
                    f"Data for key {key} must be a list or numpy array, "
                    f"but got {type(value)}."
                )
        # make sure all data is the same length
        lengths = {len(v) for v in data_ndarray.values()}
        if len(lengths) != 1:
            raise ValueError(
                "All data series in the dataset must have the same length, "
                f"but found lengths: {lengths}."
            )
        self.data = data_ndarray
        self.domain = domain or "Time [s]"
        self.control_variable = control_variable or "Current function [A]"

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
        if isinstance(value, list):
            value = np.array(value)
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

    def __contains__(self, key):
        return key in self.data

    def check(
        self, domain: str | None = None, signal: str | list[str] | None = None
    ) -> bool:
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

    def copy(self):
        """
        Create a deep copy of the Dataset instance.

        Returns
        -------
        Dataset
            A new Dataset instance with copied data, domain, and control_variable.
        """
        copied_data = copy.deepcopy(self.data)
        return Dataset(
            data_dictionary=copied_data,
            domain=self.domain,
            control_variable=self.control_variable,
        )
