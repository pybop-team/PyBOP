from pybop import BaseCost


class ExpectationPropagationCost(BaseCost):
    """
    A subclass for managing a cost that consists of several features.

    Parameters
    ----------
    costs : pybop.BaseCost
        The individual PyBOP cost objects.
    has_identical_problems : bool
        If True, the shared problem will be evaluated once and saved
        before the self.compute() method of each cost is called
        (default: False).
    has_separable_problem : bool
        This attribute must be set to False for
        ExpectationPropagationCost objects. If the corresponding
        attribute of an individual cost is True, the problem is
        separable from the cost function and will be evaluated before
        the individual cost evaluation is called.
    """

    def __init__(self, *costs):
        if not all(isinstance(cost, BaseCost) for cost in costs):
            raise TypeError("All costs must be instances of BaseCost.")
        self.costs = list(costs)
