from typing import Any
import pybop
import pybamm


class Parameter:
    """ ""
    Class for creating parameters in PyBOP.
    """

    def __init__(self, param, value=None, prior=None, bounds=None):
        self.name = param
        self.prior = prior
        self.value = value
        self.bounds = bounds

        # To Do:
        # priors implementation
        # parameter check
        # bounds checks and set defaults
        # implement methods to assign and retrieve parameters

    def __repr__(self):
        return f"Parameter: {self.name} \n Prior: {self.prior} \n Bounds: {self.bounds}"
