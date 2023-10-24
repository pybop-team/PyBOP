import pybamm

class ParameterSet:
    """
    Class for creating parameter sets in pybop.
    """

    def __new__(cls, method, name):
        if method.casefold() == "pybamm":
                return pybamm.ParameterValues(name).copy()
        else:
            raise ValueError("Only PybaMM parameter sets are currently implemented")
