from typing import Callable

import numpy as np

from pybop import Parameters
from pybop.problems.base_problem import Problem


class PythonProblem(Problem):
    """
    Defines a problem that uses a Python callable as the simulation + cost function to evaluate
    """

    def __init__(
        self,
        f: Callable = None,
        f_with_sens: Callable = None,
        params: Parameters = None,
        costs: list = None,
        cost_weights: list = None,
    ):
        super().__init__(pybop_params=params)
        if f and not callable(f):
            raise TypeError("f must be a callable function.")
        if f_with_sens and not callable(f_with_sens):
            raise TypeError("f_with_sens must be a callable function.")
        self._f = f
        self._f_with_sens = f_with_sens
        self._costs = costs
        self._cost_weights = cost_weights

    def run(self) -> float:
        """
        Evaluates the underlying simulation and cost function using the
        parameters set in the previous call to `set_params`.
        """
        self.check_set_params_called()
        sol = self._f(self.params)
        return np.dot(self._cost_weights, [sol[n].values[0] for n in self._costs])

    def run_with_sensitivities(self) -> tuple[float, np.ndarray]:
        """
        Evaluates the underlying simulation and cost function using the
        parameters set in the previous call to `set_params`, returns
        the cost and sensitivities.
        """
        self.check_set_params_called()
        self._f_with_sens(self.params)

    def set_params(self, p: np.ndarray) -> None:
        """
        Sets the parameters for the simulation and cost function.
        """
        self.check_and_store_params(p)
