from typing import Optional, Union

import numpy as np
from multivariate_parameters import MultivariateParameters

from pybop import BaseCost, BaseOptimiser, Inputs, OptimisationResult


class BaseBayesOptimiser(BaseOptimiser):
    def __init__(
        self,
        cost,
        **optimiser_kwargs,
    ):
        super().__init__(cost, **optimiser_kwargs)


class BayesianOptimisationResult(OptimisationResult):
    """
    Stores the result of a Bayesian optimisation.

    Attributes
    ----------
    x : ndarray
        The MAP (Maximum A Posteriori) of the optimisation.
    lower_bounds: ndarray
        The lower confidence parameter boundaries.
    upper_bounds: ndarray
        The upper confidence parameter boundaries.
    posterior : pybop.BasePrior
        The probability distribution of the optimisation. (PyBOP
        currently handles all probability distributions as "Priors".)
    final_cost : float
        The cost associated with the MAP ``x``.
    n_iterations : int or dict
        Number of iterations performed by the optimizer. Since Bayesian
        optimisers tend to have layers of various optimisation
        algorithms, their iteration counts may be put individually.
    optim : pybop.BaseOptimiser
        The instance of the utilized optimisation algorithm.
    time : float or dict
        The wall-clock time of the optimiser in seconds. You may give
        this as a dict to also store the processing unit time.
    """

    def __init__(
        self,
        x: Union[Inputs, np.ndarray] = None,
        lower_bounds: Union[Inputs, np.ndarray] = None,
        upper_bounds: Union[Inputs, np.ndarray] = None,
        posterior: MultivariateParameters = None,
        cost: Union[BaseCost, None] = None,
        final_cost: Optional[float] = None,
        n_iterations: Union[int, dict, None] = None,
        optim: Optional[BaseOptimiser] = None,
        time: Union[float, dict, None] = None,
    ):
        super().__init__(x, cost, final_cost, n_iterations, optim, time)
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.posterior = posterior
