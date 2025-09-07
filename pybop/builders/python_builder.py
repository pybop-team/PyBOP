from collections.abc import Callable

from pybop import PythonProblem
from pybop.builders.base import BaseBuilder


class Python(BaseBuilder):
    """
    Builder for Python-based problems.

    This builder creates problems using custom Python functions instead of specialised simulation
    frameworks. It supports both standard functions and functions that also return the sensitivities
    of the function value with respect to the input parameters.

    If this problem is used with a PyBOP optimiser or sampler, the function value will be minimised.

    Examples
    --------
    >>> builder = pybop.builders.Python()
    >>> builder.add_cost(my_model_function)
    >>> problem = builder.build()
    """

    def __init__(self):
        super().__init__()
        self._cost: Callable | None = None
        self._cost_with_sens: Callable | None = None

    def set_cost(self, cost_function: Callable) -> "Python":
        """
        Add a cost function to the problem.

        Parameters
        ----------
        cost_function : Callable
            Function with signature: cost_function(Inputs) -> float

        Raises
        ------
        TypeError
            If cost is not callable.
        """
        if not callable(cost_function):
            raise TypeError("Cost must be callable")

        self._cost = cost_function

        return self

    def set_cost_with_sens(self, cost_with_sens: Callable) -> "Python":
        """
        Add a cost function with sensitivities to the problem.

        Parameters
        ----------
        cost_with_sens : Callable
            Function with signature: cost_with_sens(Inputs) -> Tuple[float, np.ndarray]
            where the first returned element is the cost value and the second is the
            corresponding sensitivities with respect to the input parameters.

        Raises
        ------
        TypeError
            If cost_with_sens is not callable.
        """
        if not callable(cost_with_sens):
            raise TypeError("Cost with sensitivities must be callable")

        self._cost_with_sens = cost_with_sens

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
            If no functions are provided.
        """
        if not self._cost and not self._cost_with_sens:
            raise ValueError(
                "Either a cost function or cost with sensitivities must be provided"
            )

        if self._cost is None:
            # Inefficient method but useful to define a cost function
            self._cost = lambda inputs: self._cost_with_sens(inputs)[0]

        return PythonProblem(
            cost=self._cost,
            cost_with_sens=self._cost_with_sens,
            pybop_params=self.build_parameters(),
        )

    def __repr__(self) -> str:
        """Return string representation of the builder state."""
        return (
            f"Python(cost={len(self._cost)}, "
            f"cost_with_sens={len(self._cost_with_sens)})"
        )
