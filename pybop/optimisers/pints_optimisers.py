import pints

from pybop import BasePintsOptimiser


class GradientDescent(BasePintsOptimiser):
    """
    Implements a simple gradient descent optimization algorithm.

    This class extends the gradient descent optimiser from the PINTS library, designed
    to minimize a scalar function of one or more variables. Note that this optimiser
    does not support boundary constraints.

    Parameters
    ----------
    **optimiser_kwargs : optional
        Valid PINTS option keys and their values, for example:
        x0 : array_like
            Initial position from which optimisation will start.
        sigma0 : float
            The learning rate / Initial step size (default: 0.02).

    See Also
    --------
    pints.GradientDescent : The PINTS implementation this class is based on.
    """

    def __init__(self, cost, **optimiser_kwargs):
        if "sigma0" not in optimiser_kwargs.keys():
            optimiser_kwargs["sigma0"] = 0.02  # set default
        super().__init__(cost, pints.GradientDescent, **optimiser_kwargs)


class Adam(BasePintsOptimiser):
    """
    Implements the Adam optimization algorithm.

    This class extends the Adam optimiser from the PINTS library, which combines
    ideas from RMSProp and Stochastic Gradient Descent with momentum. Note that
    this optimiser does not support boundary constraints.

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

    def __init__(self, cost, **optimiser_kwargs):
        super().__init__(cost, pints.Adam, **optimiser_kwargs)


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
        super().__init__(cost, pints.IRPropMin, **optimiser_kwargs)


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
        super().__init__(cost, pints.PSO, **optimiser_kwargs)


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
        super().__init__(cost, pints.SNES, **optimiser_kwargs)


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
        super().__init__(cost, pints.XNES, **optimiser_kwargs)


class NelderMead(BasePintsOptimiser):
    """
    Implements the Nelder-Mead downhill simplex method from PINTS.

    This is a deterministic local optimiser. In most update steps it performs
    either one evaluation, or two sequential evaluations, so that it will not
    typically benefit from parallelisation.

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
        super().__init__(cost, pints.NelderMead, **optimiser_kwargs)


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
        x0 = optimiser_kwargs.pop("x0", cost.x0)
        if x0 is not None and len(x0) == 1:
            raise ValueError(
                "CMAES requires optimisation of >= 2 parameters at once. "
                + "Please choose another optimiser."
            )
        super().__init__(cost, pints.CMAES, **optimiser_kwargs)
