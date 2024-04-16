import warnings

import pybop


class Optimisation:
    """
    A class for conducting optimisation using PyBOP or PINTS optimisers.

    Parameters
    ----------
    cost : pybop.BaseCost or pints.ErrorMeasure
        An objective function to be optimised, which can be either a pybop.Cost or PINTS error measure
    optimiser : pybop.Optimiser or subclass of pybop.BaseOptimiser, optional
        An optimiser from either the PINTS or PyBOP framework to perform the optimisation (default: None).
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
        x0=None,
        optimiser=None,
        sigma0=None,
        verbose=False,
        physical_viability=True,
        allow_infeasible_solutions=True,
        **optimiser_kwargs,
    ):
        self.cost = cost
        self.x0 = x0 or cost.x0
        self.optimiser = optimiser or pybop.XNES
        self.verbose = verbose
        self.bounds = cost.bounds
        self.sigma0 = sigma0 or cost.sigma0
        self._n_parameters = cost._n_parameters
        self.physical_viability = physical_viability
        self.allow_infeasible_solutions = allow_infeasible_solutions

        # Check optimiser
        if not issubclass(self.optimiser, pybop.BaseOptimiser):
            raise ValueError("Unknown optimiser type")

        # Set whether to allow infeasible locations
        if self.cost.problem is not None and hasattr(self.cost.problem, "_model"):
            self.cost.problem._model.allow_infeasible_solutions = (
                self.allow_infeasible_solutions
            )
        else:
            # Turn off this feature as there is no model
            self.physical_viability = False
            self.allow_infeasible_solutions = False

        # Construct Optimiser
        self.optimiser = self.optimiser(
            self.x0, self.sigma0, self.bounds, **optimiser_kwargs
        )

        # Pass cost and settings to optimiser
        self.optimiser._cost_function = self.cost
        self.optimiser.verbose = self.verbose

        # Check if minimising or maximising
        if isinstance(self.cost, pybop.BaseLikelihood):
            self.cost._minimising = False
        self.optimiser._minimising = self.cost._minimising

    def run(self, **optimiser_kwargs):
        """
        Run the optimisation and return the optimised parameters and final cost.

        **optimiser_kwargs : optional
            Valid SciPy option keys and their values.

        Returns
        -------
        x : numpy.ndarray
            The best parameter set found by the optimisation.
        final_cost : float
            The final cost associated with the best parameters.
        """

        x, final_cost = self.optimiser.run(**optimiser_kwargs)

        # Store the optimised parameters
        if self.cost.problem is not None:
            self.store_optimised_parameters(x)

        # Store the log
        self.log = self.optimiser.log

        # Check if parameters are viable
        if self.physical_viability:
            self.check_optimal_parameters(x)

        return x, final_cost

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
