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

    def __init__(self, name, data):
        """
        Initialize a Dataset instance with a name and data.

        Parameters
        ----------
        name : str
            The name for the dataset.
        data : array-like
            The experimental data to store within the dataset.
        """

        self.name = name
        self.data = data

    def __repr__(self):
        """
        Return a string representation of the Dataset instance.

        Returns
        -------
        str
            A string that includes the name and data of the dataset.
        """
        return f"Dataset: {self.name} \n Data: {self.data}"

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
