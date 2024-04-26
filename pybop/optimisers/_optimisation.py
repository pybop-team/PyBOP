import warnings

import pybop

DEFAULT_OPTIMISER_OPTIONS = dict(
    x0=None,
    bounds=None,
    sigma0=0.1,
    verbose=False,
    physical_viability=True,
    allow_infeasible_solutions=True,
    _max_iterations=None,
)


class BaseOptimiser:
    """
    A base class for defining optimisation methods.

    This class serves as a base class for creating optimisers. It provides a basic structure for
    an optimisation algorithm, including the initial setup and a method stub for performing the
    optimisation process. Child classes should override set_options and the _run method with a
    specific algorithm.

    Parameters
    ----------
    cost : pybop.BaseCost or pints.ErrorMeasure
        An objective function to be optimised, which can be either a pybop.Cost
    sigma0 : float or sequence, optional
        Initial step size or standard deviation for the optimiser (default: None).
    verbose : bool, optional
        If True, the optimisation progress is printed (default: False).
    physical_viability : bool, optional
        If True, the feasibility of the optimised parameters is checked (default: True).
    allow_infeasible_solutions : bool, optional
        If True, infeasible parameter values will be allowed in the optimisation (default: True).

    Attributes
    ----------
    x0 : numpy.ndarray
        Initial parameter values for the optimisation.
    bounds : dict
        Dictionary containing the parameter bounds with keys 'lower' and 'upper'.
    _n_parameters : int
        Number of parameters in the optimisation problem.
    sigma0 : float or sequence
        Initial step size or standard deviation for the optimiser.
    log : list
        Log of the optimisation process.
    """

    def __init__(
        self,
        cost,
        **optimiser_kwargs,
    ):
        self.cost = cost
        self.__dict__.update(DEFAULT_OPTIMISER_OPTIONS)
        if isinstance(cost, pybop.BaseCost):
            self.x0 = cost.x0
            self.bounds = cost.bounds
            self.sigma0 = cost.sigma0
            self._n_parameters = cost._n_parameters
        self.log = []
        self.set_max_iterations()
        self.set_allow_infeasible_solutions()
        self.set_options(**optimiser_kwargs)

        # Check if minimising or maximising
        if isinstance(self.cost, pybop.BaseLikelihood):
            self.cost._minimising = False
        self._minimising = self.cost._minimising

    def set_options(self, **optimiser_kwargs):
        """
        Update the optimiser options and check that all have been applied.

        Parameters
        ----------
        **optimiser_kwargs : optional
            Valid option keys and their values.
        """
        # Update and remove optimiser options from the optimiser_kwargs dictionary
        # in the child class first and then in this base class
        optimiser_kwargs = self._set_options(**optimiser_kwargs)

        key_list = list(optimiser_kwargs.keys())
        for key in key_list:
            if key in ["x0", "bounds", "sigma0", "verbose"]:
                self.__dict__.update({key: optimiser_kwargs.pop(key)})
            elif key == "allow_infeasible_solutions":
                self.allow_infeasible_solutions = self.set_allow_infeasible_solutions(
                    optimiser_kwargs.pop(key)
                )
            elif key == "parallel":
                self.set_parallel(optimiser_kwargs.pop(key))

        # Throw an error if any arguments remain
        if optimiser_kwargs:
            raise ValueError(f"Unrecognised keyword arguments: {optimiser_kwargs}")

    def _set_options(self, **optimiser_kwargs):
        """
        Update the optimiser options and remove the corresponding entries from the
        optimiser_kwargs dictionary in advance of passing to the parent class
        - this function is to be overwritten by child classes.

        Parameters
        ----------
        **optimiser_kwargs : optional
            Valid option keys and their values.

        Returns
        -------
        optimiser_kwargs : dict
            Remaining option keys and their values.
        """
        return optimiser_kwargs

    def run(self, **optimiser_kwargs):
        """
        Run the optimisation and return the optimised parameters and final cost.

        Parameters
        ----------
        **optimiser_kwargs : optional
            Valid option keys and their values, for example:
            x0 : ndarray
                Initial guess for the parameters.

        Returns
        -------
        x : numpy.ndarray
            The best parameter set found by the optimisation.
        final_cost : float
            The final cost associated with the best parameters.
        """
        self.set_options(**optimiser_kwargs)

        x, final_cost = self._run()

        # Store the optimised parameters
        if self.cost.problem is not None:
            self.store_optimised_parameters(x)

        # Store the log
        self.log = self.log

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

    def set_max_iterations(self, iterations=1000):
        """
        Set the maximum number of iterations as a stopping criterion.
        Credit: PINTS

        Parameters
        ----------
        iterations : int, optional
            The maximum number of iterations to run.
            Set to `None` to remove this stopping criterion.
        """
        if iterations is not None:
            iterations = int(iterations)
            if iterations < 0:
                raise ValueError("Maximum number of iterations cannot be negative.")
        self._max_iterations = iterations

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

        if self.cost.problem is not None and hasattr(self.cost.problem, "_model"):
            self.cost.problem._model.allow_infeasible_solutions = (
                self.allow_infeasible_solutions
            )
        else:
            # Turn off this feature as there is no model
            self.physical_viability = False
            self.allow_infeasible_solutions = False


class Optimisation:
    """
    A high-level class for optimisation using PyBOP or PINTS optimisers.

    This class provides an alternative API to the `PyBOP.Optimiser()` API,
    specifically allowing for single user-friendly interface for the
    optimisation process.The class can be used with either PyBOP or PINTS
    optimisers.

    Parameters
    ----------
    cost : pybop.BaseCost or pints.ErrorMeasure
        An objective function to be optimized, which can be either a pybop.Cost
    optimiser : pybop.Optimiser or subclass of pybop.BaseOptimiser, optional
        An optimiser from either the PINTS or PyBOP framework to perform the optimization (default: None).
    sigma0 : float or sequence, optional
        Initial step size or standard deviation for the optimiser (default: None).
    verbose : bool, optional
        If True, the optimization progress is printed (default: False).
    physical_viability : bool, optional
        If True, the feasibility of the optimised parameters is checked (default: True).
    allow_infeasible_solutions : bool, optional
        If True, infeasible parameter values will be allowed in the optimisation (default: True).

    Attributes
    ----------
    All attributes from the pybop.optimiser() class

    """

    def __init__(self, cost, optimiser=None, **optimiser_kwargs):
        if optimiser is None:
            self.optimiser = pybop.DefaultOptimiser(cost, **optimiser_kwargs)
        elif issubclass(optimiser, pybop.BasePintsOptimiser):
            self.optimiser = optimiser(cost, **optimiser_kwargs)
        elif issubclass(optimiser, pybop.BaseSciPyOptimiser):
            self.optimiser = optimiser(cost, **optimiser_kwargs)
        else:
            raise ValueError("Unknown optimiser type")

    def run(self, **optimiser_kwargs):
        return self.optimiser.run(**optimiser_kwargs)

    def __getattr__(self, attr):
        # Delegate the attribute lookup to `self.optimiser`
        return getattr(self.optimiser, attr)
