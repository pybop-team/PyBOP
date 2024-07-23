from typing import Optional

import numpy as np

from pybop import BaseCost
from pybop.parameters.parameter import Inputs


class WeightedCost(BaseCost):
    """
    A subclass for constructing a linear combination of cost functions as
    a single weighted cost function.

    Inherits all parameters and attributes from ``BaseCost``.

    Additional Attributes
    ---------------------
    costs : list[pybop.BaseCost]
        A list of PyBOP cost objects.
    weights : list[float]
        A list of values with which to weight the cost values.
    _different_problems : bool
        If True, the problem for each cost is evaluated independently during
        each evaluation of the cost (default: False).
    """

    def __init__(self, *args, weights: Optional[list[float]] = None):
        self.costs = []
        for cost in args:
            if not isinstance(cost, BaseCost):
                raise TypeError(f"Received {type(cost)} instead of cost object.")
            self.costs.append(cost)
        self.weights = weights
        self._different_problems = False

        if self.weights is None:
            self.weights = np.ones(len(self.costs))
        elif isinstance(self.weights, list):
            self.weights = np.array(self.weights)
        if not isinstance(self.weights, np.ndarray):
            raise TypeError(
                "Expected a list or array of weights the same length as costs."
            )
        if not len(self.weights) == len(self.costs):
            raise ValueError(
                "Expected a list or array of weights the same length as costs."
            )

        # Check if all costs depend on the same problem
        for cost in self.costs:
            if hasattr(cost, "problem") and cost.problem is not self.costs[0].problem:
                self._different_problems = True

        if not self._different_problems:
            super().__init__(self.costs[0].problem)
            self._fixed_problem = self.costs[0]._fixed_problem
        else:
            super().__init__()
            self._fixed_problem = False
            for cost in self.costs:
                self.parameters.join(cost.parameters)

    def _evaluate(self, inputs: Inputs, grad=None):
        """
        Calculate the weighted cost for a given set of parameters.

        Parameters
        ----------
        inputs : Inputs
            The parameters for which to compute the cost.
        grad : array-like, optional
            An array to store the gradient of the cost function with respect
            to the parameters.

        Returns
        -------
        float
            The weighted cost value.
        """
        e = np.empty_like(self.costs)

        if not self._fixed_problem and self._different_problems:
            self.parameters.update(values=list(inputs.values()))
        elif not self._fixed_problem:
            self._current_prediction = self.problem.evaluate(inputs)

        for i, cost in enumerate(self.costs):
            if not self._fixed_problem and self._different_problems:
                inputs = cost.parameters.as_dict()
                cost._current_prediction = cost.problem.evaluate(inputs)
            else:
                cost._current_prediction = self._current_prediction
            e[i] = cost._evaluate(inputs, grad)

        return np.dot(e, self.weights)

    def _evaluateS1(self, inputs: Inputs):
        """
        Compute the weighted cost and its gradient with respect to the parameters.

        Parameters
        ----------
        inputs : Inputs
            The parameters for which to compute the cost and gradient.

        Returns
        -------
        tuple
            A tuple containing the cost and the gradient. The cost is a float,
            and the gradient is an array-like of the same length as `x`.
        """
        e = np.empty_like(self.costs)
        de = np.empty((len(self.parameters), len(self.costs)))

        if not self._fixed_problem and self._different_problems:
            self.parameters.update(values=list(inputs.values()))
        elif not self._fixed_problem:
            self._current_prediction, self._current_sensitivities = (
                self.problem.evaluateS1(inputs)
            )

        for i, cost in enumerate(self.costs):
            if not self._fixed_problem and self._different_problems:
                inputs = cost.parameters.as_dict()
                cost._current_prediction, cost._current_sensitivities = (
                    cost.problem.evaluateS1(inputs)
                )
            else:
                cost._current_prediction, cost._current_sensitivities = (
                    self._current_prediction,
                    self._current_sensitivities,
                )
            e[i], de[:, i] = cost._evaluateS1(inputs)

        e = np.dot(e, self.weights)
        de = np.dot(de, self.weights)

        return e, de
