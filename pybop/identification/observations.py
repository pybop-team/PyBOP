import pybamm

class Observed:
    """
    Class for experimental Observations.
    """

    def __init__(self, name, data):
        self.name = name
        self.data = data

    def __repr__(self):
        return f"Observation: {self.name} \n Data: {self.data}"

    def Interpolant(self):
        if self.variable == "time":
            self.Interpolant = pybamm.Interpolant(self.x, self.y, pybamm.t)
        else:
            NotImplementedError("Only time interpolation is supported")
