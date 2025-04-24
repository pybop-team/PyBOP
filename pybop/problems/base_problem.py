import numpy as np

from pybop import Parameters


class Problem:
    """
    Defines a callable function `f(x)` that returns the evaluation
    of a cost function. The input `x` is a set of parameters that are
    passed via the `set_params` method. The cost function is evaluated
    using the `run` method.
    """

    def __init__(self, pybop_params: Parameters = None):
        if pybop_params is None:
            self._param_names = []
        self._params = pybop_params
        self._param_names = pybop_params.keys()

    def check_and_store_params(self, p: np.ndarray) -> None:
        """
        Checks if the parameters are valid.
        """
        if len(p) != len(self._param_names):
            raise ValueError(
                f"Expected {len(self._param_names)} parameters, but got {len(p)}."
            )
        if not isinstance(p, np.ndarray):
            raise TypeError("Parameters must be a numpy array.")
        if not np.issubdtype(p.dtype, np.number):
            raise TypeError("Parameters must be a numeric numpy array.")
        self._params.update(values=p)

    def check_set_params_called(self) -> None:
        """
        Checks if the parameters have been set.
        """
        if self._params is None:
            raise ValueError(
                "Parameters have not been set. Call `set_params` before running the simulation."
            )

    @property
    def params(self) -> np.ndarray:
        """
        Returns the parameters set for the simulation and cost function.
        """
        return self._params

    @property
    def param_names(self) -> list[str]:
        """
        Returns the names of the parameters set for the simulation and cost function.
        """
        return self._param_names

    def run(self) -> float:
        """
        Evaluates the underlying simulation and cost function using the
        parameters set in the previous call to `set_params`.
        """
        raise NotImplementedError

    def run_with_sensitivities(
        self,
    ) -> tuple[float, np.ndarray]:
        """
        Evaluates the underlying simulation and cost function using the
        parameters set in the previous call to `set_params`, returns
        the cost and sensitivities.
        """
        raise NotImplementedError

    def set_params(self, p: np.ndarray) -> None:
        """
        Sets the parameters for the simulation and cost function.
        """
        raise NotImplementedError
