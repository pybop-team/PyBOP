from typing import Optional, Union

import numpy as np
from multivariate_parameters import MultivariateParameters

from pybop import BaseOptimiser, Inputs, OptimisationResult


class BaseBayesOptimiser(BaseOptimiser):
    def __init__(
        self,
        cost,
        **optimiser_kwargs,
    ):
        super().__init__(cost, **optimiser_kwargs)


class BayesianOptimisationResult(OptimisationResult):
    """
    Stores the result of a Bayesian optimisation. This only documents
    arguments changed from or added to pybop.OptimisationResult.

    Attributes
    ----------
    lower_bounds: ndarray
        The lower confidence parameter boundaries.
    upper_bounds: ndarray
        The upper confidence parameter boundaries.
    posterior : pybop.BasePrior
        The probability distribution of the optimisation. (PyBOP
        currently handles all probability distributions as "Priors".)
    n_iterations : int or dict
        Number of iterations performed by the optimiser. Since Bayesian
        optimisers tend to have layers of various optimisation
        algorithms, their iteration counts may be put individually.
    n_evaluations : int or dict
        Number of evaluations performed by the optimiser. Since Bayesian
        optimisers tend to have layers of various optimisation
        algorithms, their evaluation counts my be put individually.
    """

    def __init__(
        self,
        optim: Optional[BaseOptimiser] = None,
        x: Union[Inputs, np.ndarray] = None,
        final_cost: Optional[float] = None,
        n_iterations: Union[int, dict, None] = None,
        n_evaluations: Union[int, dict, None] = None,
        time: Union[float, dict, None] = None,
        lower_bounds: Union[Inputs, np.ndarray] = None,
        upper_bounds: Union[Inputs, np.ndarray] = None,
        posterior: MultivariateParameters = None,
    ):
        super().__init__(optim, x, final_cost, n_iterations, n_evaluations, time)
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.posterior = posterior
