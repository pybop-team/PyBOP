from collections.abc import Callable

import numpy as np

from pybop.parameters.parameter import Inputs, Parameters
from pybop.problems.base_problem import Problem


class PythonProblem(Problem):
    """
    Defines a problem that uses Python callables for simulation and cost evaluation.

    Provides a flexible interface for custom Python functions instead of relying
    on external simulation frameworks like PyBaMM. Designed to work with the Python
    builder pattern for progressive function composition.

    The problem supports standard functions and functions with sensitivities.

    Parameters
    ----------
    cost : Callable, optional
        A cost function that accepts a list of inputs and return a 1D array of cost values
        of length `len(inputs)`.
    cost_with_sens : Callable, optional
        A function that returns both the cost and sensitivities for a list of inputs. It
        should return a tuple containing a 1D array of cost values of length `len(inputs)`
        and a 2D array of sets of gradients with shape (`len(inputs)`, number of parameters).
    pybop_params : Parameters, optional
        Container for optimisation parameters defining the parameter space.

    Raises
    ------
    ValueError
        If no functions are provided.

    Examples
    --------
    >>> def quadratic_func(inputs):
    ...     return np.sum(inputs["x"]**2)
    >>>
    >>> problem = PythonProblem(
    ...     cost=quadratic_func,
    ...     pybop_params=my_params,
    ... )
    >>> result = problem.run(1.5)

    Notes
    -----
    This class can also be instantiated via the Python builder class.
    """

    def __init__(
        self,
        cost: Callable = None,
        cost_with_sens: Callable | None = None,
        vectorised: bool = False,
        pybop_params: Parameters | None = None,
    ):
        super().__init__(pybop_params=pybop_params)
        self._cost = cost
        self._cost_with_sens = cost_with_sens
        self.vectorised = vectorised

    def _compute_costs(self, inputs: list[Inputs]) -> np.ndarray:
        """
        Evaluate the function (without sensitivities) for the given inputs and return the
        cost values.

        Returns
        -------
        costs : np.ndarray
            A 1D array of function values of length `len(inputs)`.

        Raises
        ------
        RuntimeError
            If no standard functions are available (i.e., only sensitivity functions exist)
        """
        if self._cost is None:
            raise RuntimeError(
                "No standard functions configured. This problem uses sensitivity functions. "
                "Use run_with_sensitivities() instead."
            )

        if self.vectorised:
            return self._cost(inputs)

        costs = np.empty(len(inputs))
        for i, x in enumerate(inputs):
            costs[i] = float(self._cost(x))
        return costs

    def _compute_costs_and_sensitivities(
        self, inputs: list[Inputs]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate the function with sensitivities and return the cost and sensitivity values.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing the cost and parameter sensitivities:

            - costs (np.ndarray): A 1D array of function values of length `len(inputs)`.
            - sensitivities (np.ndarray): A 2D array of function values and gradients with respect
                to the input parameters with shape (`len(inputs)`, number of parameters)

        Raises
        ------
        RuntimeError
            If no sensitivity functions are available or if function evaluation fails
        """
        if self._cost_with_sens is None:
            raise RuntimeError(
                "No sensitivity functions configured. This problem uses standard functions. "
                "Use run() instead."
            )

        if self.vectorised:
            return self._cost_with_sens(inputs)

        # Pre-allocate arrays
        costs = np.empty(len(inputs))
        sens = np.empty((len(inputs), self._n_params))

        # Evaluate functions and collect results
        for i, x in enumerate(inputs):
            cost, grad = self._cost_with_sens(x)
            costs[i] = float(cost)  # Ensure scalar
            sens[i, :] = np.asarray(grad, dtype=np.float64)

        return costs, sens

    @property
    def has_sensitivities(self):
        return True if self._cost_with_sens is not None else False
