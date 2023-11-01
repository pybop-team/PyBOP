class BaseOptimiser:
    """

    Base class for the optimisation methods.

    """

    def __init__(self):
        self.name = "Base Optimiser"

    def optimise(self, cost_function, x0=None, bounds=None):
        """
        Optimisiation method to be overloaded by child classes.

        """
        self.cost_function = cost_function
        self.x0 = x0
        self.bounds = bounds

        # Run optimisation
        result = self._runoptimise(self.cost_function, self.x0, self.bounds)

        return result

    def _runoptimise(self, cost_function, x0=None, bounds=None):
        """
        Run optimisation method, to be overloaded by child classes.

        """
        pass
