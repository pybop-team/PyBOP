import pybop
import pybamm
import numpy as np


class Observed:
    """
    Class for experimental Observations.
    """

    def __init__(self, name, data):
        self.name = name
        self.data = data

    def Interpolant(self):
        if self.variable == "time":
            self.Interpolant = pybamm.Interpolant(self.x, self.y, pybamm.t)
        else:
            NotImplementedError("Only time interpolation is supported")
