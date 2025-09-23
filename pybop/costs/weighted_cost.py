import numpy as np

from pybop.costs.base_cost import BaseCost
from pybop.costs.design_cost import DesignCost


class WeightedCost(BaseCost):
    """
    A subclass for constructing a linear combination of cost functions as
    a single weighted cost function.

    Parameters
    ----------
    costs : pybop.BaseCost
        The individual PyBOP cost objects.
    weights : list[float]
        A list of values with which to weight the cost values.
    """

    def __init__(self, *costs, weights: list[float] | None = None):
        if not all(isinstance(cost, BaseCost) for cost in costs):
            raise TypeError("All costs must be instances of BaseCost.")
        if len(set(isinstance(cost, DesignCost) for cost in costs)) > 1:
            raise TypeError(
                "Costs must be either all design costs or all error measures."
            )
        self.costs = [cost for cost in costs]
        if len(set(cost.domain for cost in self.costs)) > 1:
            raise ValueError("All costs must have the same domain.")
        super().__init__()

        self.domain = self.costs[0].domain
        self.target = []
        for cost in self.costs:
            self.target.extend(cost.target)
            self.parameters.join(cost.parameters)

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

        # Apply the minimising property from each cost
        for i, cost in enumerate(self.costs):
            self.weights[i] = self.weights[i] * (1 if cost.minimising else -1)
        if all(not cost.minimising for cost in self.costs):
            # If all costs are maximising, convert the weighted cost to maximising
            self.weights = -self.weights
            self.minimising = False

    def compute(
        self,
        y: dict[str, np.ndarray],
        dy: dict | None = None,
    ) -> float | tuple[float, np.ndarray]:
        """
        Computes the cost function for the given predictions.

        Parameters
        ----------
        y : dict[str, np.ndarray[np.float64]]
            The dictionary of predictions with keys designating the output variables for fitting.
        dy : dict[str, dict[str, np.ndarray]], optional
            The corresponding sensitivities to each parameter for each output variable.

        Returns
        -------
        np.float64 or tuple[np.float64, np.ndarray[np.float64]]
            If dy is not None, returns a tuple containing the cost (float) and the
            gradient with dimension (len(parameters)), otherwise returns only the cost.
        """
        e = np.empty_like(self.costs)
        de = np.empty((len(self.parameters), len(self.costs)))

        for i, cost in enumerate(self.costs):
            if dy is not None:
                e[i], de[:, i] = cost.compute(y, dy=dy)
            else:
                e[i] = cost.compute(y)

        e = np.dot(e, self.weights)
        if dy is not None:
            de = np.dot(de, self.weights)
            return e, de

        return e
