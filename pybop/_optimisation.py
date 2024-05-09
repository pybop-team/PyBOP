import warnings

from pybop import BaseCost


class Optimisation:
    """
    A base class for defining optimisation methods.

    This class serves as a base class for creating optimisers. It provides a basic structure for
    an optimisation algorithm, including the initial setup and a method stub for performing the
    optimisation process. Child classes should override _set_up_optimiser and the _run method with
    a specific algorithm.

    Parameters
    ----------
    cost : pybop.BaseCost or pints.ErrorMeasure
        An objective function to be optimised, which can be either a pybop.Cost or PINTS error measure
    **optimiser_kwargs : optional
            Valid option keys and their values.

    Attributes
    ----------
    x0 : numpy.ndarray
        Initial parameter values for the optimisation.
    bounds : dict
        Dictionary containing the parameter bounds with keys 'lower' and 'upper'.
    sigma0 : float or sequence
        Initial step size or standard deviation for the optimiser.
    verbose : bool, optional
        If True, the optimisation progress is printed (default: False).
    physical_viability : bool, optional
        If True, the feasibility of the optimised parameters is checked (default: True).
    allow_infeasible_solutions : bool, optional
        If True, infeasible parameter values will be allowed in the optimisation (default: True).
    log : list
        A log of the parameter values tried during the optimisation.
    """

    def __init__(
        self,
        cost,
        **optimiser_kwargs,
    ):
        # First set attributes to default values
        self.x0 = None
        self.bounds = None
        self.sigma0 = 0.1
        self.verbose = False
        self.log = []
        self._minimising = True
        self.physical_viability = False
        self.allow_infeasible_solutions = False
        self.default_max_iterations = 1000
        self.result = None

        if isinstance(cost, BaseCost):
            self.cost = cost
            self.x0 = cost.x0
            self.bounds = cost.bounds
            self.sigma0 = cost.sigma0
            self._minimising = cost._minimising
            self.set_allow_infeasible_solutions()
        else:
            warnings.warn(
                "The cost is not an instance of pybop.BaseCost. Continuing "
                + "under the assumption that it is a callable function.",
                UserWarning,
            )
            self.cost = BaseCost()

            def cost_evaluate(x, grad=None):
                return cost(x)

            self.cost._evaluate = cost_evaluate

        self.unset_options = optimiser_kwargs
        self.set_base_options()
        self._set_up_optimiser()

        # Throw an error if any options remain
        if self.unset_options:
            raise ValueError(f"Unrecognised keyword arguments: {self.unset_options}")

    def set_base_options(self):
        """
        Update the base optimiser options and remove them from the options dictionary.
        """
        key_list = list(self.unset_options.keys())
        for key in key_list:
            if key in ["x0", "bounds", "sigma0", "verbose"]:
                self.__dict__.update({key: self.unset_options.pop(key)})
            elif key == "allow_infeasible_solutions":
                self.set_allow_infeasible_solutions(self.unset_options.pop(key))

    def _set_up_optimiser(self):
        """
        Parse optimiser options and prepare the optimiser.

        This method should be implemented by child classes.

        Raises
        ------
        NotImplementedError
            If the method has not been implemented by the subclass.
        """
        raise NotImplementedError

    def run(self):
        """
        Run the optimisation and return the optimised parameters and final cost.

        Returns
        -------
        x : numpy.ndarray
            The best parameter set found by the optimisation.
        final_cost : float
            The final cost associated with the best parameters.
        """
        x, final_cost = self._run()

        # Store the optimised parameters
        if hasattr(self.cost, "parameters"):
            self.store_optimised_parameters(x)

        # Check if parameters are viable
        if self.physical_viability:
            self.check_optimal_parameters(x)

        return x, final_cost

    def _run(self):
        """
        Contains the logic for the optimisation algorithm.

        This method should be implemented by child classes to perform the actual optimisation.

        Raises
        ------
        NotImplementedError
            If the method has not been implemented by the subclass.
        """
        raise NotImplementedError

    def store_optimised_parameters(self, x):
        """
        Update the problem parameters with optimised values.

        The optimised parameter values are stored within the associated PyBOP parameter class.

        Parameters
        ----------
        x : array-like
            Optimised parameter values.
        """
        for i, param in enumerate(self.cost.parameters):
            param.update(value=x[i])

    def check_optimal_parameters(self, x):
        """
        Check if the optimised parameters are physically viable.
        """

        if self.cost.problem._model.check_params(
            inputs=x, allow_infeasible_solutions=False
        ):
            return
        else:
            warnings.warn(
                "Optimised parameters are not physically viable! \nConsider retrying the optimisation"
                + " with a non-gradient-based optimiser and the option allow_infeasible_solutions=False",
                UserWarning,
                stacklevel=2,
            )

    def name(self):
        """
        Returns the name of the optimiser, to be overwritten by child classes.

        Returns
        -------
        str
            The name of the optimiser, which is "Optimisation" for this base class.
        """
        return "Optimisation"

    def set_allow_infeasible_solutions(self, allow=True):
        """
        Set whether to allow infeasible solutions or not.

        Parameters
        ----------
        iterations : bool, optional
            Whether to allow infeasible solutions.
        """
        # Set whether to allow infeasible locations
        self.physical_viability = allow
        self.allow_infeasible_solutions = allow

        if hasattr(self.cost, "problem") and hasattr(self.cost.problem, "_model"):
            self.cost.problem._model.allow_infeasible_solutions = (
                self.allow_infeasible_solutions
            )
        else:
            # Turn off this feature as there is no model
            self.physical_viability = False
            self.allow_infeasible_solutions = False
