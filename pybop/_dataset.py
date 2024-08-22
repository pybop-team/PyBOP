from typing import Union

import numpy as np
from pybamm import solvers


class Dataset:
    """
    Represents a collection of experimental observations.

    This class provides a structured way to store and work with experimental data,
    which may include applying operations such as interpolation.

    Parameters
    ----------
    data_dictionary : dict or instance of pybamm.solvers.solution.Solution
        The experimental data to store within the dataset.
    """

    def __init__(self, data_dictionary):
        """
        Initialize a Dataset instance with data and a set of names.
        """

        if isinstance(data_dictionary, solvers.solution.Solution):
            data_dictionary = data_dictionary.get_data_dict()
        if not isinstance(data_dictionary, dict):
            raise TypeError("The input to pybop.Dataset must be a dictionary.")
        self.data = data_dictionary
        self.names = self.data.keys()

    def __repr__(self):
        """
        Return a string representation of the Dataset instance.

        Returns
        -------
        str
            A string that includes the type and contents of the dataset.
        """
        return f"Dataset: {type(self.data)} \n Contains: {self.names}"

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

    def check(self, domain: str = None, signal: Union[str, list[str]] = None) -> bool:
        """
        Check the consistency of a PyBOP Dataset against the expected format.

        Parameters
        ----------
        domain : str, optional
            The domain of the dataset. Defaults to "Time [s]".
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
        self.domain = domain or "Time [s]"
        signals = [signal] if isinstance(signal, str) else (signal or ["Voltage [V]"])

        # Check that the dataset contains domain and chosen signals
        missing_attributes = set([self.domain, *signals]) - set(self.names)
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
