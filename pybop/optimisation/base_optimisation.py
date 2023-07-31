import pybop


class BaseOptimisation(object):
    """

    Base class for the optimisation methods.

    """

    def __init__(self):
        self.name = "Base Optimisation"

    def optimise(self, cost_function, method=None, x0=None, bounds=None, options=None):
        """
        Optimise method to be overloaded by child classes.

        """
        # Set up optimisation
        self.cost_function = cost_function
        self.x0 = x0 or cost_function.x0
        self.options = options
        self.method = method or cost_function.default_method
        self.bounds = bounds or cost_function.bounds

        # Run optimisation
        result = self._runoptimise(
            cost_function, self.method, self.x0, self.bounds, self.options
        )

        return result

    def _runoptimise(self, cost_function, method, x0, bounds, options):
        """
        Run optimisation method, to be overloaded by child classes.

        """
        pass
