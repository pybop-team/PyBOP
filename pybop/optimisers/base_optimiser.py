class BaseOptimiser:
    """
    A base class for defining optimisation methods.

    This class serves as a template for creating optimisers. It provides a basic structure for
    an optimisation algorithm, including the initial setup and a method stub for performing the
    optimisation process. Child classes should override update_options and the_run method with
    a specific algorithm.
    """

    def __init__(self, x0=None, sigma0=None, bounds=None):
        """
        Initialises the BaseOptimiser.

        Parameters
        ----------
        x0 : array_like
            Initial position from which optimisation will start.
        sigma0 : float, optional
            Initial step size or standard deviation depending on the optimiser.
        bounds : sequence or Bounds, optional
            Bounds on the parameters (default: None).
        """
        self._cost_function = None
        self.x0 = x0
        self.sigma0 = sigma0
        self.bounds = bounds

        self.log = []
        self._max_iterations = None
        self.set_max_iterations()

    def update_options(self, **optimiser_kwargs):
        """
        Update the optimiser options, to be overwritten by child classes.

        Parameters
        ----------
        **optimiser_kwargs : optional
            Valid option keys and their values.
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

    def run(self, x0=None, **optimiser_kwargs):
        """
        Initiates the optimisation process.

        Parameters
        ----------
        x0 : ndarray, optional
            Initial guess for the parameters (default: None).
        **optimiser_kwargs : optional
            Valid option keys and their values.

        Returns
        -------
        The result of the optimisation process. The specific type of this result will depend on the child implementation.
        """
        if x0 is not None:
            self.x0 = x0

        # Run optimisation
        result = self._run(**optimiser_kwargs)

        return result

    def _run(self, **optimiser_kwargs):
        """
        Contains the logic for the optimisation algorithm.

        This method should be implemented by child classes to perform the actual optimisation.

        Parameters
        ----------
        **optimiser_kwargs : optional
            Valid option keys and their values.

        Raises
        ------
        NotImplementedError
            If the method has not been implemented by the subclass.
        """
        raise NotImplementedError

    def set_max_iterations(self, iterations=1000):
        """
        Set the maximum number of iterations as a stopping criterion.
        Credit: PINTS

        Parameters
        ----------
        iterations : int, optional
            The maximum number of iterations to run (default: 1000).
            Set to `None` to remove this stopping criterion.
        """
        if iterations is not None:
            iterations = int(iterations)
            if iterations < 0:
                raise ValueError("Maximum number of iterations cannot be negative.")
        self._max_iterations = iterations
