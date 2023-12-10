class BaseOptimiser:
    """

    Base class for the optimisation methods.

    """

    def __init__(self):
        pass

    def optimise(self, cost_function, x0=None, bounds=None, maxiter=None):
        """
        Optimisiation method to be overloaded by child classes.

        """
        self.cost_function = cost_function
        self.x0 = x0
        self.bounds = bounds
        self.maxiter = maxiter

        # Run optimisation
        result = self._runoptimise(self.cost_function, x0=self.x0, bounds=self.bounds)

        return result

    def _runoptimise(self, cost_function, x0=None, bounds=None):
        """
        Run optimisation method, to be overloaded by child classes.

        """
        pass

    def name(self):
        """
        Returns the name of the optimiser.
        """
        return "Base Optimiser"
