class BaseOptimisation:
    """

    Base class for the optimisation methods.

    """

    def __init__(self):
        self.name = "Base Optimisation"

    def optimise(self, cost_function, x0, bounds, method=None):
        """
        Optimise method to be overloaded by child classes.

        """
        # Set up optimisation
        self.cost_function = cost_function
        self.x0 = x0
        self.method = method
        self.bounds = bounds

        # Run optimisation
        result = self._runoptimise(self.cost_function, self.x0, self.bounds)

        return result

    def _runoptimise(self, cost_function, x0, bounds):
        """
        Run optimisation method, to be overloaded by child classes.

        """
        pass
