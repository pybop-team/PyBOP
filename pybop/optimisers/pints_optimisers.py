from warnings import warn

from pints import CMAES as PintsCMAES
from pints import PSO as PintsPSO
from pints import SNES as PintsSNES
from pints import XNES as PintsXNES
from pints import Adam as PintsAdam
from pints import GradientDescent as PintsGradientDescent
from pints import IRPropMin as PintsIRPropMin
from pints import NelderMead as PintsNelderMead

from pybop import AdamWImpl, BasePintsOptimiser, CuckooSearchImpl


class GradientDescent(BasePintsOptimiser):
    """
    Implements a simple gradient descent optimization algorithm.

    This class extends the gradient descent optimiser from the PINTS library, designed
    to minimize a scalar function of one or more variables.

    Note that this optimiser does not support boundary constraints.

    Parameters
    ----------
    **optimiser_kwargs : optional
        Valid PINTS option keys and their values, for example:
        x0 : array_like
            Initial position from which optimisation will start.
        sigma0 : float
            The learning rate / Initial step size.

    See Also
    --------
    pints.GradientDescent : The PINTS implementation this class is based on.
    """

    def __init__(self, cost, **optimiser_kwargs):
        super().__init__(cost, PintsGradientDescent, **optimiser_kwargs)


class Adam(BasePintsOptimiser):
    """
    Implements the Adam optimization algorithm.

    This class extends the Adam optimiser from the PINTS library, which combines
    ideas from RMSProp and Stochastic Gradient Descent with momentum.

    Note that this optimiser does not support boundary constraints.

    Parameters
    ----------
    **optimiser_kwargs : optional
        Valid PINTS option keys and their values, for example:
        x0 : array_like
            Initial position from which optimisation will start.
        sigma0 : float
            Initial step size.

    See Also
    --------
    pints.Adam : The PINTS implementation this class is based on.
    """

    warn(
        "Adam is deprecated and will be removed in a future release. Please use AdamW instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    def __init__(self, cost, **optimiser_kwargs):
        super().__init__(cost, PintsAdam, **optimiser_kwargs)


class AdamW(BasePintsOptimiser):
    """
    Implements the AdamW optimisation algorithm in PyBOP.

    This class extends the AdamW optimiser, which is a variant of the Adam
    optimiser that incorporates weight decay. AdamW is designed to be more
    robust and stable for training deep neural networks, particularly when
    using larger learning rates.

    Note that this optimiser does not support boundary constraints.
    Parameters
    ----------
    **optimiser_kwargs : optional
        Valid PyBOP option keys and their values, for example:
        x0 : array_like
            Initial position from which optimisation will start.
        sigma0 : float
            Initial step size.

    See Also
    --------
    pybop.AdamWImpl : The PyBOP implementation this class is based on.
    """

    def __init__(self, cost, **optimiser_kwargs):
        super().__init__(cost, AdamWImpl, **optimiser_kwargs)


class IRPropMin(BasePintsOptimiser):
    """
    Implements the iRpropMin optimization algorithm.

    This class inherits from the PINTS IRPropMin class, which is an optimiser that
    uses resilient backpropagation with weight-backtracking. It is designed to handle
    problems with large plateaus, noisy gradients, and local minima.

    Parameters
    ----------
    **optimiser_kwargs : optional
        Valid PINTS option keys and their values, for example:
        x0 : array_like
            Initial position from which optimisation will start.
        sigma0 : float
            Initial step size.
        bounds : dict
            A dictionary with 'lower' and 'upper' keys containing arrays for lower and
            upper bounds on the parameters.

    See Also
    --------
    pints.IRPropMin : The PINTS implementation this class is based on.
    """

    def __init__(self, cost, **optimiser_kwargs):
        super().__init__(cost, PintsIRPropMin, **optimiser_kwargs)


class PSO(BasePintsOptimiser):
    """
    Implements a particle swarm optimization (PSO) algorithm.

    This class extends the PSO optimiser from the PINTS library. PSO is a
    metaheuristic optimization method inspired by the social behavior of birds
    flocking or fish schooling, suitable for global optimization problems.

    Parameters
    ----------
    **optimiser_kwargs : optional
        Valid PINTS option keys and their values, for example:
        x0 : array_like
            Initial positions of particles, which the optimisation will use.
        sigma0 : float
            Spread of the initial particle positions.
        bounds : dict
            A dictionary with 'lower' and 'upper' keys containing arrays for lower and
            upper bounds on the parameters.

    See Also
    --------
    pints.PSO : The PINTS implementation this class is based on.
    """

    def __init__(self, cost, **optimiser_kwargs):
        super().__init__(cost, PintsPSO, **optimiser_kwargs)


class SNES(BasePintsOptimiser):
    """
    Implements the stochastic natural evolution strategy (SNES) optimization algorithm.

    Inheriting from the PINTS SNES class, this optimiser is an evolutionary algorithm
    that evolves a probability distribution on the parameter space, guiding the search
    for the optimum based on the natural gradient of expected fitness.

    Parameters
    ----------
    **optimiser_kwargs : optional
        Valid PINTS option keys and their values, for example:
        x0 : array_like
            Initial position from which optimisation will start.
        sigma0 : float
            Initial standard deviation of the sampling distribution.
        bounds : dict
            A dictionary with 'lower' and 'upper' keys containing arrays for lower and
            upper bounds on the parameters.

    See Also
    --------
    pints.SNES : The PINTS implementation this class is based on.
    """

    def __init__(self, cost, **optimiser_kwargs):
        super().__init__(cost, PintsSNES, **optimiser_kwargs)


class XNES(BasePintsOptimiser):
    """
    Implements the Exponential Natural Evolution Strategy (XNES) optimiser from PINTS.

    XNES is an evolutionary algorithm that samples from a multivariate normal
    distribution, which is updated iteratively to fit the distribution of successful
    solutions.

    Parameters
    ----------
    **optimiser_kwargs : optional
        Valid PINTS option keys and their values, for example:
        x0 : array_like
            The initial parameter vector to optimise.
        sigma0 : float
            Initial standard deviation of the sampling distribution.
        bounds : dict
            A dictionary with 'lower' and 'upper' keys containing arrays for lower and
            upperbounds on the parameters. If ``None``, no bounds are enforced.

    See Also
    --------
    pints.XNES : PINTS implementation of XNES algorithm.
    """

    def __init__(self, cost, **optimiser_kwargs):
        super().__init__(cost, PintsXNES, **optimiser_kwargs)


class NelderMead(BasePintsOptimiser):
    """
    Implements the Nelder-Mead downhill simplex method from PINTS.

    This is a deterministic local optimiser. In most update steps it performs
    either one evaluation, or two sequential evaluations, so that it will not
    typically benefit from parallelisation.

    Note that this optimiser does not support boundary constraints.

    Parameters
    ----------
    **optimiser_kwargs : optional
        Valid PINTS option keys and their values, for example:
        x0 : array_like
            The initial parameter vector to optimise.
        sigma0 : float
            Initial standard deviation of the sampling distribution.
            Does not appear to be used.

    See Also
    --------
    pints.NelderMead : PINTS implementation of Nelder-Mead algorithm.
    """

    def __init__(self, cost, **optimiser_kwargs):
        super().__init__(cost, PintsNelderMead, **optimiser_kwargs)


class CMAES(BasePintsOptimiser):
    """
    Adapter for the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) optimiser in PINTS.

    CMA-ES is an evolutionary algorithm for difficult non-linear non-convex optimization problems.
    It adapts the covariance matrix of a multivariate normal distribution to capture the shape of
    the cost landscape.

    Parameters
    ----------
    **optimiser_kwargs : optional
        Valid PINTS option keys and their values, for example:
        x0 : array_like
            The initial parameter vector to optimise.
        sigma0 : float
            Initial standard deviation of the sampling distribution.
        bounds : dict
            A dictionary with 'lower' and 'upper' keys containing arrays for lower and
            upper bounds on the parameters. If ``None``, no bounds are enforced.

    See Also
    --------
    pints.CMAES : PINTS implementation of CMA-ES algorithm.
    """

    def __init__(self, cost, **optimiser_kwargs):
        x0 = optimiser_kwargs.get("x0", cost.parameters.initial_value())
        if len(x0) == 1 or len(cost.parameters) == 1:
            raise ValueError(
                "CMAES requires optimisation of >= 2 parameters at once. "
                "Please choose another optimiser."
            )
        super().__init__(cost, PintsCMAES, **optimiser_kwargs)


class CuckooSearch(BasePintsOptimiser):
    """
    Adapter for the Cuckoo Search optimiser in PyBOP.

    Cuckoo Search is a population-based optimisation algorithm inspired by the brood parasitism of some cuckoo species.
    It is designed to be simple, efficient, and robust, and is suitable for global optimisation problems.

    Parameters
    ----------
    **optimiser_kwargs : optional
        Valid PyBOP option keys and their values, for example:
        x0 : array_like
            Initial parameter values.
        sigma0 : float
            Initial step size.
        bounds : dict
            A dictionary with 'lower' and 'upper' keys containing arrays for lower and
            upper bounds on the parameters.

    See Also
    --------
    pybop.CuckooSearch : PyBOP implementation of Cuckoo Search algorithm.
    """

    def __init__(self, cost, **optimiser_kwargs):
        super().__init__(cost, CuckooSearchImpl, **optimiser_kwargs)
