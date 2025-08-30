from collections.abc import Callable, Sequence

import numpy as np

from pybop import Parameters
from pybop.problems.base_problem import Problem


class PythonProblem(Problem):
    """
    Defines a problem that uses Python callables for simulation and cost evaluation.

    Provides a flexible interface for custom Python functions instead of relying
    on external simulation frameworks like PyBaMM. Designed to work with the Python
    builder pattern for progressive function composition.

    The problem supports either standard funcs OR funcs with sensitivities, but not both
    simultaneously.

    Parameters
    ----------
    func : Sequence[Callable], optional
        Sequence of functions that perform simulation. Each function should accept
        parameter values and return a numeric result or dict of results.
    func_with_sens : Sequence[Callable], optional
        Sequence of functions that perform simulation and return sensitivities.
        Each function should accept parameter values and return (results, gradients).
    pybop_params : Parameters, optional
        Container for optimisation parameters defining the parameter space.
    weights : Sequence[float], optional
        Weights for each callable component. Length must match the number of functions.

    Raises
    ------
    ValueError
        If both function types are provided, if neither is provided, or if weights
        length doesn't match function count.

    Examples
    --------
    >>> def quadratic_func(params):
    ...     return np.sum(params**2)
    >>>
    >>> problem = PythonProblem(
    ...     func=[quadratic_func],
    ...     pybop_params=my_params,
    ...     weights=[1.0]
    ... )
    >>> result = problem.run()

    Notes
    -----
    This class is typically instantiated via the Python builder class rather
    than directly, which provides a more convenient fluent interface.
    """

    def __init__(
        self,
        funs: Sequence[Callable] | None = None,
        funs_with_sens: Sequence[Callable] | None = None,
        pybop_params: Parameters | None = None,
        weights: Sequence[float] | None = None,
    ):
        super().__init__(pybop_params=pybop_params)
        self._funs = tuple(funs) if funs is not None else None
        self._funs_with_sens = (
            tuple(funs_with_sens) if funs_with_sens is not None else None
        )
        self._has_sensitivities = True if funs_with_sens is not None else False
        self._weights = np.asarray(weights) if weights is not None else None

    def run(self) -> np.ndarray:
        """
        Execute all standard functions with current parameters and return weighted sum.

        This method evaluates each function with the current parameter values,
        applies the corresponding weights, and returns the sum.

        Returns
        -------
        np.ndarray
            value (np.ndarray): Weighted sum of function values as a 1D array

        Raises
        ------
        RuntimeError
            If no standard functions are available (i.e., only sensitivity functions exist)
        """

        if self._funs is None:
            raise RuntimeError(
                "No standard functions configured. This problem uses sensitivity functions. "
                "Use run_with_sensitivities() instead."
            )

        xs = self.params.get_values().T
        try:
            if xs.ndim == 1:
                results = np.asarray([func(xs) for func in self._funs])
                cost = np.dot(self._weights, results)
            else:
                cost = []
                for x in xs:
                    results = np.asarray([func(x) for func in self._funs])
                    cost.append(np.dot(self._weights, results))
        except (TypeError, ValueError) as e:
            raise RuntimeError(f"function evaluation failed: {e}") from e

        return cost

    def run_with_sensitivities(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Execute all sensitivity functions and return weighted results with gradients.

        This method evaluates each sensitivity function, which returns both
        values and gradients, then computes weighted sums for both components.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing the cost and parameter sensitivities:

            - value (np.ndarray): Weighted sum of function values as a 1D array
            - sensitivities (np.ndarray): Weighted sum of parameter gradients
              with shape (n_params,)

        Raises
        ------
        RuntimeError
            If no sensitivity functions are available or if function evaluation fails
        """
        if self._funs_with_sens is None:
            raise RuntimeError(
                "No sensitivity functions configured. This problem uses standard functions. "
                "Use run() instead."
            )

        # Pre-allocate arrays
        n_funs = len(self._funs_with_sens)
        values = np.empty(n_funs, dtype=np.float64)
        gradients = []

        # Evaluate funcs and collect results
        try:
            for i, func in enumerate(self._funs_with_sens):
                val, grad = func(self.params.get_values())
                values[i] = float(val)  # Ensure scalar
                gradients.append(np.asarray(grad, dtype=np.float64))
        except (TypeError, ValueError, AttributeError) as e:
            raise RuntimeError(f"Sensitivity function evaluation failed: {e}") from e

        # Compute weighted results
        weighted_value = np.dot(values, self._weights[:, np.newaxis])

        # Stack and weight gradients
        if gradients:
            grad_matrix = np.stack(gradients, axis=0)
            weighted_gradient = np.dot(self._weights, grad_matrix)
        else:
            weighted_gradient = np.array([])

        return weighted_value, weighted_gradient

    def set_params(self, p: np.ndarray) -> None:
        """
        Sets the parameter values for simulation.

        Parameters
        ----------
        p : np.ndarray
            Array of parameter values
        """
        self.check_and_store_params(p)
