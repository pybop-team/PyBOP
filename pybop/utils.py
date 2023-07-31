import pybop
import pybamm


class Interpolant:
    def __init__(self, x, y, variable):
        self.name = "Interpolant"
        if variable == "time":
            self.Interpolant = pybamm.Interpolant(x, y, pybamm.t)
        else:
            NotImplementedError("Only time interpolation is supported")
