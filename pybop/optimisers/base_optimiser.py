class BaseOptimiser:
    """
    A base class for defining optimisation methods.

    This class serves as a template for creating optimisers. It provides a basic structure for
    an optimisation algorithm, including the initial setup and a method stub for performing
    the optimisation process. Child classes should override the optimise and _runoptimise
    methods with specific algorithms.
    """

    def __init__(self, bounds=None):
        """
        Initializes the BaseOptimiser.

        Parameters
        ----------
        bounds : sequence or Bounds, optional
            Bounds on the parameters. Default is None.
        """
        self.bounds = bounds
        self._max_iterations = None

    def optimise(self, cost_function, x0=None, **optimiser_kwargs):
        """
        Initiates the optimisation process.

        Parameters
        ----------
        cost_function : callable
            The cost function to be minimised by the optimiser.
        x0 : ndarray, optional
            Initial guess for the parameters. Default is None.

        Returns
        -------
        The result of the optimisation process. The specific type of this result will depend on the child implementation.
        """
        self.cost_function = cost_function
        self.x0 = x0

        # Run optimisation
        result = self._runoptimise(self.cost_function, x0=self.x0, **optimiser_kwargs)

        return result

    def _runoptimise(self, cost_function, x0=None, **optimiser_kwargs):
        """
        Contains the logic for the optimisation algorithm.

        This method should be implemented by child classes to perform the actual optimisation.

        Parameters
        ----------
        cost_function : callable
            The cost function to be minimised by the optimiser.
        x0 : ndarray, optional
            Initial guess for the parameters. Default is None.

        Returns
        -------
        This method is expected to return the result of the optimisation, the format of which
        will be determined by the child class implementation.
        """
        pass

    def name(self):
        """
        Returns the name of the optimiser, to be overwritten by child classes.

        Returns
        -------
        str
            The name of the optimiser, which is "BaseOptimiser" for this base class.
        """
        return "BaseOptimiser"
