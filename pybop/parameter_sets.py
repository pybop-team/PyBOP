import pybamm
import pybop

class ParameterSet:
    """
    Class for creating parameter sets in pybop.
    """

    def __new__(cls, method, name):
        if method.casefold() == "pybamm":
            try:
                return pybamm.ParameterValues(name).copy()
            except:
                raise ValueError("Parameter set not found")
        else:
            raise ValueError("Only PybaMM parameter sets are currently implemented")