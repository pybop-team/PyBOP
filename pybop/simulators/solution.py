import numpy as np

from pybop.parameters.parameter import Inputs


class SolutionVariable:
    """
    A class to store the simulation results for one variable.
    """

    def __init__(
        self, data: np.ndarray, sensitivities: dict[str, np.ndarray] | None = None
    ):
        self.data = data
        self.sensitivities = sensitivities


class Solution:
    """
    A class to store simulation results, inspired by pybamm.Solution.
    """

    def __init__(self, inputs: Inputs = None):
        self._dict = {}
        self.all_inputs = [inputs] if inputs is not None else []

    def set_solution_variable(
        self,
        variable_name: str,
        data: np.ndarray,
        sensitivities: dict[str, np.ndarray] | None = None,
    ):
        self._dict[variable_name] = SolutionVariable(
            data=data, sensitivities=sensitivities
        )

    def __getitem__(self, key):
        return self._dict[key]
