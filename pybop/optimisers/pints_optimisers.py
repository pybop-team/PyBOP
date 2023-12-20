import pints


class GradientDescent(pints.GradientDescent):
    """
    Implements a simple gradient descent optimization algorithm.

    This class extends the gradient descent optimiser from the PINTS library, designed
    to minimize a scalar function of one or more variables. Note that this optimiser
    does not support boundary constraints.

    Parameters
    ----------
    x0 : array_like
        Initial position from which optimization will start.
    sigma0 : float, optional
        Initial step size (default is 0.1).
    bounds : sequence or ``Bounds``, optional
        Ignored by this optimiser, provided for API consistency.

    See Also
    --------
    pints.GradientDescent : The PINTS implementation this class is based on.
    """

    def __init__(self, x0, sigma0=0.1, bounds=None):
        if bounds is not None:
            print("NOTE: Boundaries ignored by Gradient Descent")

        self.boundaries = None  # Bounds ignored in pints.GradDesc
        super().__init__(x0, sigma0, self.boundaries)


class Adam(pints.Adam):
    """
    Implements the Adam optimization algorithm.

    This class extends the Adam optimiser from the PINTS library, which combines
    ideas from RMSProp and Stochastic Gradient Descent with momentum. Note that
    this optimiser does not support boundary constraints.

    Parameters
    ----------
    x0 : array_like
        Initial position from which optimization will start.
    sigma0 : float, optional
        Initial step size (default is 0.1).
    bounds : sequence or ``Bounds``, optional
        Ignored by this optimiser, provided for API consistency.

    See Also
    --------
    pints.Adam : The PINTS implementation this class is based on.
    """

    def __init__(self, x0, sigma0=0.1, bounds=None):
        if bounds is not None:
            print("NOTE: Boundaries ignored by Adam")

        self.boundaries = None  # Bounds ignored in pints.Adam
        super().__init__(x0, sigma0, self.boundaries)


class IRPropMin(pints.IRPropMin):
    """
    Implements the iRpropMin optimization algorithm.

    This class inherits from the PINTS IRPropMin class, which is an optimiser that
    uses resilient backpropagation with weight-backtracking. It is designed to handle
    problems with large plateaus, noisy gradients, and local minima.

    Parameters
    ----------
    x0 : array_like
        Initial position from which optimization will start.
    sigma0 : float, optional
        Initial step size (default is 0.1).
    bounds : dict, optional
        Lower and upper bounds for each optimization parameter.

    See Also
    --------
    pints.IRPropMin : The PINTS implementation this class is based on.
    """

    def __init__(self, x0, sigma0=0.1, bounds=None):
        if bounds is not None:
            self.boundaries = pints.RectangularBoundaries(
                bounds["lower"], bounds["upper"]
            )
        else:
            self.boundaries = None
        super().__init__(x0, sigma0, self.boundaries)


class PSO(pints.PSO):
    """
    Implements a particle swarm optimization (PSO) algorithm.

    This class extends the PSO optimiser from the PINTS library. PSO is a
    metaheuristic optimization method inspired by the social behavior of birds
    flocking or fish schooling, suitable for global optimization problems.

    Parameters
    ----------
    x0 : array_like
        Initial positions of particles, which the optimization will use.
    sigma0 : float, optional
        Spread of the initial particle positions (default is 0.1).
    bounds : dict, optional
        Lower and upper bounds for each optimization parameter.

    See Also
    --------
    pints.PSO : The PINTS implementation this class is based on.
    """

    def __init__(self, x0, sigma0=0.1, bounds=None):
        if bounds is not None:
            self.boundaries = pints.RectangularBoundaries(
                bounds["lower"], bounds["upper"]
            )
        else:
            self.boundaries = None
        super().__init__(x0, sigma0, self.boundaries)


class SNES(pints.SNES):
    """
    Implements the stochastic natural evolution strategy (SNES) optimization algorithm.

    Inheriting from the PINTS SNES class, this optimiser is an evolutionary algorithm
    that evolves a probability distribution on the parameter space, guiding the search
    for the optimum based on the natural gradient of expected fitness.

    Parameters
    ----------
    x0 : array_like
        Initial position from which optimization will start.
    sigma0 : float, optional
        Initial step size (default is 0.1).
    bounds : dict, optional
        Lower and upper bounds for each optimization parameter.

    See Also
    --------
    pints.SNES : The PINTS implementation this class is based on.
    """

    def __init__(self, x0, sigma0=0.1, bounds=None):
        if bounds is not None:
            self.boundaries = pints.RectangularBoundaries(
                bounds["lower"], bounds["upper"]
            )
        else:
            self.boundaries = None
        super().__init__(x0, sigma0, self.boundaries)


class XNES(pints.XNES):
    """
    Implements the Exponential Natural Evolution Strategy (XNES) optimiser from PINTS.

    XNES is an evolutionary algorithm that samples from a multivariate normal distribution, which is updated iteratively to fit the distribution of successful solutions.

    Parameters
    ----------
    x0 : array_like
        The initial parameter vector to optimize.
    sigma0 : float, optional
        Initial standard deviation of the sampling distribution, defaults to 0.1.
    bounds : dict, optional
        A dictionary with 'lower' and 'upper' keys containing arrays for lower and upper bounds on the parameters. If ``None``, no bounds are enforced.

    See Also
    --------
    pints.XNES : PINTS implementation of XNES algorithm.
    """

    def __init__(self, x0, sigma0=0.1, bounds=None):
        if bounds is not None:
            self.boundaries = pints.RectangularBoundaries(
                bounds["lower"], bounds["upper"]
            )
        else:
            self.boundaries = None
        super().__init__(x0, sigma0, self.boundaries)


class CMAES(pints.CMAES):
    """
    Adapter for the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) optimiser in PINTS.

    CMA-ES is an evolutionary algorithm for difficult non-linear non-convex optimization problems.
    It adapts the covariance matrix of a multivariate normal distribution to capture the shape of the cost landscape.

    Parameters
    ----------
    x0 : array_like
        The initial parameter vector to optimize.
    sigma0 : float, optional
        Initial standard deviation of the sampling distribution, defaults to 0.1.
    bounds : dict, optional
        A dictionary with 'lower' and 'upper' keys containing arrays for lower and upper bounds on the parameters.
        If ``None``, no bounds are enforced.

    See Also
    --------
    pints.CMAES : PINTS implementation of CMA-ES algorithm.
    """

    def __init__(self, x0, sigma0=0.1, bounds=None):
        if bounds is not None:
            self.boundaries = pints.RectangularBoundaries(
                bounds["lower"], bounds["upper"]
            )
        else:
            self.boundaries = None

        super().__init__(x0, sigma0, self.boundaries)
