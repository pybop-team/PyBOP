import pybamm


class Dataset:
    """
    Represents a collection of experimental observations.

    This class provides a structured way to store and work with experimental data,
    which may include applying operations such as interpolation.

    Parameters
    ----------
    name : str
        The name of the dataset, providing a label for identification.
    data : array-like
        The actual experimental data, typically in a structured form such as
        a NumPy array or a pandas DataFrame.

    """

    def __init__(self, data_dictionary):
        """
        Initialize a Dataset instance with a name and data.

        Parameters
        ----------
        data_dictionary : dict or instance of pybamm.solvers.solution.Solution
            The experimental data to store within the dataset.
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
