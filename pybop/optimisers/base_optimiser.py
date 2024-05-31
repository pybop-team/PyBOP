import warnings

import numpy as np

from pybop import BaseCost, BaseLikelihood, DesignCost


class BaseOptimiser:
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
    minimising : bool, optional
        If True, the target is to minimise the cost, else target is to maximise by minimising
        the negative cost (default: True).
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
        self.minimising = True
        self.physical_viability = False
        self.allow_infeasible_solutions = False
        self.default_max_iterations = 1000
        self.result = None

        if isinstance(cost, BaseCost):
            self.cost = cost
            self.x0 = cost.x0
            self.bounds = cost.bounds
            self.sigma0 = cost.sigma0
            self.set_allow_infeasible_solutions()
            if isinstance(cost, (BaseLikelihood, DesignCost)):
                self.minimising = False
        else:
            try:
                cost_test = cost(optimiser_kwargs.get("x0", []))
                warnings.warn(
                    "The cost is not an instance of pybop.BaseCost, but let's continue "
                    + "assuming that it is a callable function to be minimised.",
                    UserWarning,
                )
                self.cost = cost
                self.minimising = True

            except Exception:
                raise Exception("The cost is not a recognised cost object or function.")

            if not np.isscalar(cost_test) or not np.isreal(cost_test):
                raise TypeError(
                    f"Cost returned {type(cost_test)}, not a scalar numeric value."
                )

        self.unset_options = optimiser_kwargs
        self.set_base_options()
        self._set_up_optimiser()

        # Throw an warning if any options remain
        if self.unset_options:
            warnings.warn(
                f"Unrecognised keyword arguments: {self.unset_options} will not be used.",
                UserWarning,
                stacklevel=2,
            )

    def set_base_options(self):
        """
        Update the base optimiser options and remove them from the options dictionary.
        """
        self.x0 = self.unset_options.pop("x0", self.x0)
        self.bounds = self.unset_options.pop("bounds", self.bounds)
        self.sigma0 = self.unset_options.pop("sigma0", self.sigma0)
        self.verbose = self.unset_options.pop("verbose", self.verbose)
        self.minimising = self.unset_options.pop("minimising", self.minimising)
        if "allow_infeasible_solutions" in self.unset_options.keys():
            self.set_allow_infeasible_solutions(
                self.unset_options.pop("allow_infeasible_solutions")
            )

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
