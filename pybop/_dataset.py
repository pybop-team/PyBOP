import numpy as np
import pybamm


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

        if isinstance(data_dictionary, pybamm.solvers.solution.Solution):
            data_dictionary = data_dictionary.get_data_dict()
        if not isinstance(data_dictionary, (dict, Dataset)):
            raise ValueError("The input to pybop.Dataset must be a dictionary.")
        self.data = data_dictionary
        self.names = self.data.keys()
        self.n_datasets = 1

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
            The data series corresonding to the key.

        Raises
        ------
        ValueError
            The key must exist in the dataset.
        """
        if key not in self.data.keys():
            raise ValueError(f"The key {key} does not exist in this dataset.")

        return self.data[key]

    def Interpolant(self):
        """
        Create an interpolation function of the dataset based on the independent variable.

        Currently, only time-based interpolation is supported. This method modifies
        the instance's Interpolant attribute to be an interpolation function that
        can be evaluated at different points in time.

        Raises
        ------
        NotImplementedError
            If the independent variable for interpolation is not supported.
        """

        if self.variable == "time":
            self.Interpolant = pybamm.Interpolant(self.x, self.y, pybamm.t)
        else:
            NotImplementedError("Only time interpolation is supported")

    def check(self, signal=["Voltage [V]"]):
        """
        Check the consistency of a PyBOP Dataset against the expected format.

        Returns
        -------
        bool
            If True, the dataset has the expected attributes.

        Raises
        ------
        ValueError
            If the time series and the data series are not consistent.
        """
        if isinstance(signal, str):
            signal = [signal]

        # Check that the dataset contains time and chosen signal
        for name in ["Time [s]"] + signal:
            if name not in self.names:
                raise ValueError(f"expected {name} in list of dataset")

        # Check for increasing times
        time_data = self.data["Time [s]"]
        if np.any(time_data < 0):
            raise ValueError("Times can not be negative.")
        if np.any(time_data[:-1] >= time_data[1:]):
            raise ValueError("Times must be increasing.")

        # Check for consistent data
        n_time_data = len(time_data)
        for s in signal:
            if len(self.data[s]) != n_time_data:
                raise ValueError(f"Time data and {s} data must be the same length.")

        return True


class DatasetCollection:
    """
    Represents a collection of Datasets. Provides a simple way to
    handle multiple sets of experimental data.

    Parameters
    ----------
    datasets: list
        The datasets to store within the dataset collection.
        Individual datasets can either be a dict, or a Dataset
        instance.
    """

    def __init__(self, datasets):
        """
        Initialize a DatasetCollection instance with either a list of
        Dataset objects, or a list of dictionaries to be turned into
        Datasets
        """
        self.n_datasets = len(datasets)
        try:
            self.datasets = [Dataset(data) for data in datasets]
        except AttributeError:
            self.datasets = datasets
        self.names = set().union(*(data.names for data in self.datasets))
        self.data = {k: [data[k] for data in datasets] for k in self.names}

    def check(self, signal=["Voltage [V]"]):
        """
        Check the consistency of each PyBOP Dataset against the
        expected format.

        Returns
        -------
        bool
            If True, the dataset has the expected attributes.

        Raises
        ------
        ValueError
            If the time series and the data series are not consistent.
        """
        for dataset in self.datasets:
            dataset.check(signal)
        return True

    def __len__(self):
        return self.n_datasets

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
            The data series corresonding to the key.

        Raises
        ------
        ValueError
            The key must exist in the dataset.
        """
        if key not in self.data.keys():
            raise ValueError(f"The key {key} does not exist in this dataset.")

        return self.data[key]

    def __iter__(self):
        """
        DatasetCollection can be iterated over, to get each individual
        dataset comprising the collection.
        """
        self.__iter_index = 0
        return self

    def __next__(self):
        """
        DatasetCollection can be iterated over, to get each individual
        dataset comprising the collection.
        """
        if self.__iter_index < len(self.datasets):
            value = self.datasets[self.__iter_index]
            self.__iter_index += 1
            return value
        else:
            raise StopIteration
