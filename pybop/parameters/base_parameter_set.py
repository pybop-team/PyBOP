import pybamm


class ParameterSet:
    """
    Class for creating parameter sets in PyBOP.
    """

    def __new__(cls, method, name):
        if method.casefold() == "pybamm":
            return pybamm.ParameterValues(name).copy()
        else:
            raise ValueError("Only PyBaMM parameter sets are currently implemented")
