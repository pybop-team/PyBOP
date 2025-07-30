from collections.abc import Callable

from pybop import PythonProblem
from pybop.builders.base import BaseBuilder


class Python(BaseBuilder):
    """
    Builder for Python-based problems.

    This builder creates problems using custom Python functions instead of
    specialised simulation frameworks. It supports both standard functions and
    functions with sensitivity analysis.

    If this problem is used by Pybop Optimisation or Sampling classes, the
    functions will be minimised.

    Examples
    --------
    >>> builder = pybop.builders.Python()
    >>> builder.add_fun(my_model_function, weight=1.5)
    >>> problem = builder.build()
    """

    def __init__(self):
        super().__init__()
        self._funs: list[Callable] = []
        self._funs_with_sens: list[Callable] = []
        self._weights: list[float] = []

    def add_fun(self, model: Callable, weight: float = 1.0) -> "Python":
        """
        Add a simulation function to the problem.

        Parameters
        ----------
        model : Callable
            Function with signature: func(params: np.ndarray) -> float
        weight : float, default=1.0
            Weight for this model in multi-objective optimisation.

        Returns
        -------
        Python
            Self for method chaining.

        Raises
        ------
        TypeError
            If model is not callable.
        """
        if not callable(model):
            raise TypeError("Model must be callable")

        self._funs.append(model)
        self._weights.append(weight)
        return self

    def add_fun_with_sens(
        self, model_with_sens: Callable, weight: float = 1.0
    ) -> "Python":
        """
        Add a simulation function with sensitivity analysis to the problem.

        Parameters
        ----------
        model_with_sens : Callable
            Function with signature: func(params: np.ndarray) -> Tuple[float, np.ndarray]
            where the first returned element is the callable value and the second is the
            corresponding parameter sensitivities wrt the callable value.
        weight : float, default=1.0
            Weight for this model in multi-objective optimisation.

        Returns
        -------
        Python
            Self for method chaining.

        Raises
        ------
        TypeError
            If model_with_sens is not callable.
        """
        if not callable(model_with_sens):
            raise TypeError("Model with sensitivities must be callable")

        self._funs_with_sens.append(model_with_sens)
        self._weights.append(weight)
        return self

    def build(self) -> PythonProblem:
        """
        Build the Python problem.

        Returns
        -------
        PythonProblem
            The constructed problem with all configured components.

        Raises
        ------
        ValueError
            If no functions are provided or if both model types are specified.
        """
        if not self._funs and not self._funs_with_sens:
            raise ValueError("At least one model function must be provided")

        if self._funs and self._funs_with_sens:
            raise ValueError(
                "Cannot specify both standard functions and functions with sensitivities"
            )

        return PythonProblem(
            funs=self._funs or None,
            funs_with_sens=self._funs_with_sens or None,
            pybop_params=self.build_parameters(),
            weights=self._weights,
        )

    def __repr__(self) -> str:
        """Return string representation of the builder state."""
        return (
            f"Python(func={len(self._funs)}, "
            f"funcs_with_sens={len(self._funs_with_sens)}, "
            f"weights={len(self._weights)})"
        )
