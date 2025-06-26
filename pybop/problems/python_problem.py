from collections.abc import Callable, Sequence

import numpy as np

from pybop import Parameters
from pybop.problems.base_problem import Problem


class PythonProblem(Problem):
    """
    Defines a problem that uses Python callables for simulation and cost evaluation.

    Provides a flexible interface for custom Python functions instead of relying
    on external simulation frameworks like PyBaMM. Designed to work with the Python
    builder pattern for progressive model composition.

    The problem supports either standard models OR models with sensitivities, but not both
    simultaneously.

    Parameters
    ----------
    model : Sequence[Callable], optional
        Sequence of functions that perform simulation. Each function should accept
        parameter values and return a numeric result or dict of results.
    model_with_sens : Sequence[Callable], optional
        Sequence of functions that perform simulation and return sensitivities.
        Each function should accept parameter values and return (results, gradients).
    pybop_params : Parameters, optional
        Container for optimisation parameters defining the parameter space.
    weights : Sequence[float], optional
        Weights for each callable component. Length must match the number of models.

    Raises
    ------
    ValueError
        If both model types are provided, if neither is provided, or if weights
        length doesn't match model count.

    Examples
    --------
    >>> def quadratic_model(params):
    ...     return np.sum(params**2)
    >>>
    >>> problem = PythonProblem(
    ...     model=[quadratic_model],
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
        model: Sequence[Callable] | None = None,
        model_with_sens: Sequence[Callable] | None = None,
        pybop_params: Parameters | None = None,
        weights: Sequence[float] | None = None,
    ):
        super().__init__(pybop_params=pybop_params)
        self._model = tuple(model) if model is not None else None
        self._model_with_sens = (
            tuple(model_with_sens) if model_with_sens is not None else None
        )
        self._weights = weights

    def run(self) -> float:
        """
        Execute all standard models with current parameters and return weighted sum.

        This method evaluates each model function with the current parameter values,
        applies the corresponding weights, and returns the sum.

        Returns
        -------
        float
            Weighted sum of all model results

        Raises
        ------
        RuntimeError
            If no standard models are available (i.e., only sensitivity models exist)
        """

        if self._model is None:
            raise RuntimeError(
                "No standard models configured. This problem uses sensitivity models. "
                "Use run_with_sensitivities() instead."
            )

        # Vectorised evaluation
        try:
            results = np.fromiter(
                (model(self.params.get_values()) for model in self._model),
                dtype=np.float64,
                count=len(self._model),
            )
        except (TypeError, ValueError) as e:
            raise RuntimeError(f"Model evaluation failed: {e}") from e

        return np.dot(results, self._weights)

    def run_with_sensitivities(self) -> tuple[float, np.ndarray]:
        """
        Execute all sensitivity models and return weighted results with gradients.

        This method evaluates each sensitivity model function, which returns both
        values and gradients, then computes weighted sums for both components.

        Returns
        -------
        tuple[float, np.ndarray]
            Tuple containing:
            - Weighted sum of model values (float)
            - Weighted sum of parameter gradients (np.ndarray)

        Raises
        ------
        RuntimeError
            If no sensitivity models are available or if model evaluation fails
        """
        if self._model_with_sens is None:
            raise RuntimeError(
                "No sensitivity models configured. This problem uses standard models. "
                "Use run() instead."
            )

        # Pre-allocate arrays
        n_models = len(self._model_with_sens)
        values = np.empty(n_models, dtype=np.float64)
        gradients = []

        # Evaluate models and collect results
        try:
            for i, model in enumerate(self._model_with_sens):
                val, grad = model(self.params.get_values())
                values[i] = float(val)  # Ensure scalar
                gradients.append(np.asarray(grad, dtype=np.float64))
        except (TypeError, ValueError, AttributeError) as e:
            raise RuntimeError(f"Sensitivity model evaluation failed: {e}") from e

        # Compute weighted results
        weighted_value = np.dot(values, self._weights)

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
