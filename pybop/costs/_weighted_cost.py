import copy
from typing import Optional

import numpy as np

from pybop import BaseCost, BaseLikelihood, DesignCost
from pybop.parameters.parameter import Inputs


class WeightedCost(BaseCost):
    """
    A subclass for constructing a linear combination of cost functions as
    a single weighted cost function.

    Inherits all parameters and attributes from ``BaseCost``.

    Attributes
    ---------------------
    costs : list[pybop.BaseCost]
        A list of PyBOP cost objects.
    weights : list[float]
        A list of values with which to weight the cost values.
    _has_different_problems : bool
        If True, the problem for each cost is evaluated independently during
        each evaluation of the cost (default: False).
    """

    def __init__(self, *costs, weights: Optional[list[float]] = None):
        if not all(isinstance(cost, BaseCost) for cost in costs):
            raise TypeError("All costs must be instances of BaseCost.")
        self.costs = [copy.copy(cost) for cost in costs]
        self._has_different_problems = False
        self.minimising = not any(
            isinstance(cost, (BaseLikelihood, DesignCost)) for cost in self.costs
        )
        if len(set(type(cost.problem) for cost in self.costs)) > 1:
            raise TypeError("All problems must be of the same class type.")

        # Check if weights are provided
        if weights is not None:
            try:
                self.weights = np.asarray(weights, dtype=float)
            except ValueError:
                raise ValueError("Weights must be numeric values.") from None

            if self.weights.size != len(self.costs):
                raise ValueError("Number of weights must match number of costs.")
        else:
            self.weights = np.ones(len(self.costs))

        # Check if all costs depend on the same problem
        self._has_different_problems = any(
            hasattr(cost, "problem") and cost.problem is not self.costs[0].problem
            for cost in self.costs[1:]
        )

        if self._has_different_problems:
            super().__init__()
            for cost in self.costs:
                self.parameters.join(cost.parameters)
        else:
            super().__init__(self.costs[0].problem)
            self._predict = False
            for cost in self.costs:
                cost._predict = False

        # Check if any cost function requires capacity update
        if any(cost.update_capacity for cost in self.costs):
            self.update_capacity = True

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

        if not self._predict:
            if self._has_different_problems:
                self.parameters.update(values=list(inputs.values()))
            else:
                self.y = self.problem.evaluate(
                    inputs, update_capacity=self.update_capacity
                )

        for i, cost in enumerate(self.costs):
            if not self._has_different_problems:
                cost.y = self.y
            e[i] = cost.evaluate(inputs)

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

        if not self._predict:
            if self._has_different_problems:
                self.parameters.update(values=list(inputs.values()))
            else:
                self.y, self.dy = self.problem.evaluateS1(inputs)

        for i, cost in enumerate(self.costs):
            if not self._has_different_problems:
                cost.y, cost.dy = (self.y, self.dy)
            e[i], de[:, i] = cost.evaluateS1(inputs)

        e = np.dot(e, self.weights)
        de = np.dot(de, self.weights)

        return e, de
