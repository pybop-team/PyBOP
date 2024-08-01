import warnings
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
    _has_identical_problems : bool
        If True, the shared problem will be evaluated once and saved before the
        self._evaluate() method of each cost is called (default: False).
    _has_seperable_problem: bool
        If True, the shared problem is seperable from the cost function and
        will be evaluated for each problem before the cost evaluation is
        called (default: False).
    """

    def __init__(self, *costs, weights: Optional[list[float]] = None):
        if not all(isinstance(cost, BaseCost) for cost in costs):
            raise TypeError("All costs must be instances of BaseCost.")
        self.costs = [cost for cost in costs]
        if len(set(type(cost.problem) for cost in self.costs)) > 1:
            raise TypeError("All problems must be of the same class type.")
        self.minimising = not any(
            isinstance(cost, (BaseLikelihood, DesignCost)) for cost in self.costs
        )

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
        self._has_identical_problems = all(
            cost._has_separable_problem and cost.problem is self.costs[0].problem
            for cost in self.costs
        )

        if self._has_identical_problems:
            super().__init__(self.costs[0].problem)
        else:
            super().__init__()
            for cost in self.costs:
                self.parameters.join(cost.parameters)

        # Check if any cost function requires capacity update
        self.update_capacity = False
        if any(cost.update_capacity for cost in self.costs):
            self.update_capacity = True

            warnings.warn(
                "WeightedCost doesn't currently support DesignCosts with different `update_capacity` attributes,\n"
                f"Using global `DesignCost.update_capacity` attribute as: {self.update_capacity}",
                UserWarning,
                stacklevel=2,
            )

        # Weighted costs do not use this functionality
        self._has_separable_problem = False

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
        self.parameters.update(values=list(inputs.values()))

        if self._has_identical_problems:
            self.y = self.problem.evaluate(inputs, update_capacity=self.update_capacity)

        e = np.empty_like(self.costs)

        for i, cost in enumerate(self.costs):
            inputs = cost.parameters.as_dict()
            if self._has_identical_problems:
                cost.y = self.y
            elif cost._has_separable_problem:
                cost.y = cost.problem.evaluate(
                    inputs, update_capacity=self.update_capacity
                )
            e[i] = cost._evaluate(inputs)

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
        self.parameters.update(values=list(inputs.values()))

        if self._has_identical_problems:
            self.y, self.dy = self.problem.evaluateS1(inputs)

        e = np.empty_like(self.costs)
        de = np.empty((len(self.parameters), len(self.costs)))

        for i, cost in enumerate(self.costs):
            inputs = cost.parameters.as_dict()
            if self._has_identical_problems:
                cost.y, cost.dy = (self.y, self.dy)
            elif cost._has_separable_problem:
                cost.y, cost.dy = cost.problem.evaluateS1(inputs)
            e[i], de[:, i] = cost._evaluateS1(inputs)

        e = np.dot(e, self.weights)
        de = np.dot(de, self.weights)

        return e, de

    @property
    def has_identical_problems(self):
        return self._has_identical_problems
