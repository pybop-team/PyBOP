import numpy as np

from pybop.costs.base_cost import BaseCost
from pybop.costs.design_cost import DesignCost
from pybop.costs.evaluation import Evaluation
from pybop.parameters.parameter import Inputs
from pybop.processing.dataset import Dataset
from pybop.simulators.solution import Solution


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

        self._domain = self.costs[0].domain
        for cost in self.costs:
            self.parameters.join(cost.parameters)
        target_dataset = self.costs[0]._dataset  # noqa: SLF001
        self.set_target(
            [cost.target for cost in self.costs],
            dataset=None
            if target_dataset is None
            else Dataset(target_dataset, domain=self.costs[0].domain),
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

        # Apply the minimising property from each cost
        for i, cost in enumerate(self.costs):
            self.weights[i] = self.weights[i] * (1 if cost.minimising else -1)
        if all(not cost.minimising for cost in self.costs):
            # If all costs are maximising, convert the weighted cost to maximising
            self.weights = -self.weights
            self.minimising = False

    def evaluate_batch(
        self,
        solution: list[Solution],
        inputs: list[Inputs],
        calculate_sensitivities: bool = False,
    ) -> Evaluation:
        """
        Computes the cost function for the given predictions.

        Parameters
        ----------
        solution : list[Solution]
            A list of simulation results.
        inputs : list[Inputs]
            The corresponding list of input parameters.
        calculate_sensitivities : bool
            Whether to also return the sensitivities (default: False).
        """
        # Preallocate the evaluation results
        weighted_evaluation = Evaluation()
        weighted_evaluation.preallocate(
            inputs=inputs, calculate_sensitivities=calculate_sensitivities
        )

        for j, (sol, x) in enumerate(zip(solution, inputs, strict=False)):
            e = np.empty(len(self.costs))
            de = (
                {key: np.zeros(len(self.costs)) for key in x.keys()}
                if calculate_sensitivities
                else None
            )

            for i, cost in enumerate(self.costs):
                evaluation = cost.evaluate(
                    sol, inputs=x, calculate_sensitivities=calculate_sensitivities
                )
                e[i] = evaluation.values.item()
                if calculate_sensitivities:
                    for key, value in evaluation.sensitivities.items():
                        de[key][i] = value.item()

            # Sum with weighting
            e = np.dot(e, self.weights)
            if calculate_sensitivities:
                for key in de.keys():
                    de[key] = np.dot(de[key], self.weights)

            weighted_evaluation.insert_result(i=j, value=e, sensitivities=de)

        return weighted_evaluation

    def set_target(
        self,
        target: list[list[str]] | list[str] | str | None = None,
        dataset: Dataset | None = None,
    ):
        """Set the target variable for all costs. Expecting a list of list[str] the same length as self.costs."""
        target = [target] if isinstance(target, str) else target or self._target
        if isinstance(target[0], str):
            target = [target] * len(self.costs)

        self._target = []
        for i, cost in enumerate(self.costs):
            cost.set_target(target[i], dataset=dataset)
            self._target.extend(cost.target)

        super().set_target(target=self.target, dataset=dataset)

    @property
    def target(self):
        return list(set(self._target))
