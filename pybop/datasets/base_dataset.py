import pybamm


class Dataset:
    """
    Class for experimental observations.
    """

    def __init__(self, data_dictionary):
        if isinstance(data_dictionary, pybamm.solvers.solution.Solution):
            data_dictionary = data_dictionary.get_data_dict()
        if not isinstance(data_dictionary, dict):
            raise ValueError("The input to pybop.Dataset must be a dictionary.")
        self.data = data_dictionary
        self.names = self.data.keys()

    def __repr__(self):
        return f"Dataset: {type(self.data)} \n Contains: {self.names}"

    def Interpolant(self):
        if self.variable == "time":
            self.Interpolant = pybamm.Interpolant(self.x, self.y, pybamm.t)
        else:
            NotImplementedError("Only time interpolation is supported")
