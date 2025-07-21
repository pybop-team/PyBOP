import numpy as np

from pybop import Parameters
from pybop.problems.base_problem import Problem
import pybop_diffsol

import numpy as np


class DiffsolProblem(Problem):
    """

    Parameters
    ----------
    code : str
        The code to be executed for the problem.
    cost_type : pybop_diffsol.DiffsolCostType
        The type of cost function to be used in the problem.
    times : np.ndarray
        The time points at which the data is sampled.
    data : np.ndarray
        The data series corresponding to the time points.
    config : pybop_diffsol.Config, optional
        Configuration for the Diffsol solver. Defaults to None.
    pybop_params : Parameters, optional
        Parameters for the problem. Defaults to None.
    """

    def __init__(
        self,
        code: str,
        cost_type: pybop_diffsol.CostType,
        times: np.ndarray,
        data: np.ndarray,
        config: pybop_diffsol.Config | None = None,
        pybop_params: Parameters | None = None,
    ):
        super().__init__(pybop_params=pybop_params)
        config = config or pybop_diffsol.Config()
        is_sparse = False
        print("Using DiffsolProblem with code:", code)
        if is_sparse:
            self._diffsol = pybop_diffsol.DiffsolSparse(code, config)
        else:
            self._diffsol = pybop_diffsol.DiffsolDense(code, config)
        self._data = data
        self._times = times
        self._cost_type = cost_type

    def run(self) -> float:
        return self._diffsol.cost(self._times, self._data, self._cost_type)

    def run_with_sensitivities(self) -> tuple[float, np.ndarray]:
        return self._diffsol.sens(self._times, self._data, self._cost_type)

    def set_params(self, p: np.ndarray) -> None:
        """
        Sets the parameter values for simulation.

        Parameters
        ----------
        p : np.ndarray
            Array of parameter values
        """
        self.check_and_store_params(p)
        self._diffsol.set_params(p)
