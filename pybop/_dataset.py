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
        if not isinstance(data_dictionary, dict):
            raise ValueError("The input to pybop.Dataset must be a dictionary.")
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
        """
        if isinstance(signal, str):
            signal = [signal]

        # Check that the dataset contains time and chosen signal
        for name in ["Time [s]"] + signal:
            if name not in self.names:
                raise ValueError(f"expected {name} in list of dataset")

        # Check for consistent data
        for s in signal:
            if len(self.data[s]) != len(self.data["Time [s]"]):
                raise ValueError(f"Time data and {s} data must be the same length.")

        return True
