import pybop
from typing import Callable
import numpy as np


class PythonProblem(pybop.Problem):
    """
    Defines a problem that uses a Python callable as the simulation + cost function to evaluate
    """

    def __init__(
        self, f: Callable, f_with_sens: Callable, param_names: list[str] = None
    ):
        super().__init__(param_names=param_names)
        if not callable(f):
            raise TypeError("f must be a callable function.")
        if not callable(f_with_sens):
            raise TypeError("f_with_sens must be a callable function.")
        self._f = f
        self._f_with_sens = f_with_sens

    def run(self) -> float:
        """
        Evaluates the underlying simulation and cost function using the
        parameters set in the previous call to `set_params`.
        """
        self.check_set_params_called()
        return self._f(self.params)

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
        self.check_params(p)
        self._params = p
