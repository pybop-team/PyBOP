from typing import Optional, Union

import numpy as np

from pybop import BaseCost, BaseLikelihood, DesignCost


class WeightedCost(BaseCost):
    """
    A subclass for constructing a linear combination of cost functions as
    a single weighted cost function.

    Inherits all parameters and attributes from ``BaseCost``.

    Attributes
    ---------------------
    costs : pybop.BaseCost
        The individual PyBOP cost objects.
    weights : list[float]
        A list of values with which to weight the cost values.
    has_identical_problems : bool
        If True, the shared problem will be evaluated once and saved before the
        self.compute() method of each cost is called (default: False).
    has_separable_problem: bool
        This attribute must be set to False for WeightedCost objects. If the
        corresponding attribute of an individual cost is True, the problem is
        separable from the cost function and will be evaluated before the
        individual cost evaluation is called.
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
            cost.has_separable_problem and cost.problem is self.costs[0].problem
            for cost in self.costs
        )

        if self._has_identical_problems:
            super().__init__(self.costs[0].problem)
        else:
            super().__init__()

        for cost in self.costs:
            self.join_parameters(cost.parameters)

        # Weighted costs do not use this functionality
        self._has_separable_problem = False

    def compute(
        self,
        y: dict,
        dy: np.ndarray = None,
        calculate_grad: bool = False,
    ) -> Union[float, tuple[float, np.ndarray]]:
        """
        Computes the cost function for the given predictions.

        Parameters
        ----------
        y : dict
            The dictionary of predictions with keys designating the signals for fitting.
        dy : np.ndarray, optional
            The corresponding gradient with respect to the parameters for each signal.
        calculate_grad : bool, optional
            A bool condition designating whether to calculate the gradient.

        Returns
        -------
        float
            The weighted cost value.
        """
        if self._has_identical_problems:
            inputs = self.problem.parameters.as_dict()
            if calculate_grad:
                y, dy = self.problem.evaluateS1(inputs)
            else:
                y = self.problem.evaluate(inputs)

        e = np.empty_like(self.costs)
        de = np.empty((len(self.parameters), len(self.costs)))

        for i, cost in enumerate(self.costs):
            inputs = cost.parameters.as_dict()
            if self._has_identical_problems:
                y, dy = (y, dy)
            elif cost.has_separable_problem:
                if calculate_grad:
                    y, dy = cost.problem.evaluateS1(inputs)
                else:
                    y = cost.problem.evaluate(inputs)

            if calculate_grad:
                e[i], de[:, i] = cost.compute(y, dy=dy, calculate_grad=True)
            else:
                e[i] = cost.compute(y)

        e = np.dot(e, self.weights)
        if calculate_grad:
            de = np.dot(de, self.weights)
            return e, de

        return e

    @property
    def has_identical_problems(self):
        return self._has_identical_problems
