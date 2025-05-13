from typing import Optional
from pints import CMAES as PintsCMAES
from pints import PSO as PintsPSO
from pints import SNES as PintsSNES
from pints import XNES as PintsXNES
from pints import IRPropMin as PintsIRPropMin
from pints import NelderMead as PintsNelderMead

from pybop.problems.base_problem import Problem
import pybop
from pybop import (
    AdamWImpl,
    BasePintsOptimiser,
    CuckooSearchImpl,
    GradientDescentImpl,
    IRPropPlusImpl,
    RandomSearchImpl,
    SimulatedAnnealingImpl,
)


class GradientDescent(BasePintsOptimiser):
    """
    Implements a simple gradient descent optimisation algorithm.

    This class extends the gradient descent optimiser from the PINTS library, designed
    to minimise a scalar function of one or more variables.

    Note that this optimiser does not support boundary constraints.

    Parameters
    ----------
    problem: pybop.Problem
        The cost function to be minimised.
    options: pybop.PintsOptions
        optimisation options

    See Also
    --------
    pints.GradientDescent : The PINTS implementation this class is based on.
    """

    def __init__(
        self,
        problem: Problem,
        options: Optional[pybop.PintsOptions],
    ):
        super().__init__(
            problem,
            GradientDescentImpl,
            options,
        )


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
    problem: pybop.Problem
        The cost function to be minimised.
    options: pybop.PintsOptions
        optimisation options

    See Also
    --------
    pybop.AdamWImpl : The PyBOP implementation this class is based on.
    """

    def __init__(
        self,
        problem: Problem,
        options: Optional[pybop.PintsOptions],
    ):
        super().__init__(
            problem,
            AdamWImpl,
            options,
        )


class IRPropMin(BasePintsOptimiser):
    """
    Implements the iRpropMin optimisation algorithm.

    This class inherits from the PINTS IRPropMin class, which is an optimiser that
    uses resilient backpropagation without weight-backtracking. It is designed to handle
    problems with large plateaus, noisy gradients, and local minima.

    Parameters
    ----------
    problem: pybop.Problem
        The cost function to be minimised.
    options: pybop.PintsOptions
        optimisation options

    See Also
    --------
    pints.IRPropMin : The PINTS implementation this class is based on.
    """

    def __init__(
        self,
        problem: Problem,
        options: Optional[pybop.PintsOptions],
    ):
        super().__init__(
            problem,
            PintsIRPropMin,
            options,
        )


class IRPropPlus(BasePintsOptimiser):
    """
    Implements the iRpropPlus optimisation algorithm.

    This class implements the improved resilient backpropagation with weight-backtracking.
    It is designed to handle problems with large plateaus, noisy gradients, and local minima.

    Parameters
    ----------
    problem: pybop.Problem
        The cost function to be minimised.
    options: pybop.PintsOptions
        optimisation options

    See Also
    --------
    pints.IRPropMin : The PINTS implementation this class is based on.
    """

    def __init__(
        self,
        problem: Problem,
        options: Optional[pybop.PintsOptions],
    ):
        super().__init__(
            problem,
            IRPropPlusImpl,
            options,
        )


class PSO(BasePintsOptimiser):
    """
    Implements a particle swarm optimisation (PSO) algorithm.

    This class extends the PSO optimiser from the PINTS library. PSO is a
    metaheuristic optimisation method inspired by the social behavior of birds
    flocking or fish schooling, suitable for global optimisation problems.

    Parameters
    ----------
    problem: pybop.Problem
        The cost function to be minimised.
    options: pybop.PintsOptions
        optimisation options

    See Also
    --------
    pints.PSO : The PINTS implementation this class is based on.
    """

    def __init__(
        self,
        problem: Problem,
        options: Optional[pybop.PintsOptions],
    ):
        super().__init__(
            problem,
            PintsPSO,
            options,
        )


class SNES(BasePintsOptimiser):
    """
    Implements the stochastic natural evolution strategy (SNES) optimisation algorithm.

    Inheriting from the PINTS SNES class, this optimiser is an evolutionary algorithm
    that evolves a probability distribution on the parameter space, guiding the search
    for the optimum based on the natural gradient of expected fitness.

    Parameters
    ----------
    problem: pybop.Problem
        The cost function to be minimised.
    options: pybop.PintsOptions
        optimisation options

    See Also
    --------
    pints.SNES : The PINTS implementation this class is based on.
    """

    def __init__(
        self,
        problem: Problem,
        options: Optional[pybop.PintsOptions],
    ):
        super().__init__(
            problem,
            PintsSNES,
            options,
        )


class XNES(BasePintsOptimiser):
    """
    Implements the Exponential Natural Evolution Strategy (XNES) optimiser from PINTS.

    XNES is an evolutionary algorithm that samples from a multivariate normal
    distribution, which is updated iteratively to fit the distribution of successful
    solutions.

    Parameters
    ----------
    problem: pybop.Problem
        the cost function to be minimised.
    options: pybop.PintsOptions
        optimisation options

    See Also
    --------
    pints.XNES : PINTS implementation of XNES algorithm.
    """

    def __init__(
        self,
        problem: Problem,
        options: Optional[pybop.PintsOptions],
    ):
        super().__init__(
            problem,
            PintsXNES,
            options,
        )


class NelderMead(BasePintsOptimiser):
    """
    Implements the Nelder-Mead downhill simplex method from PINTS.

    This is a deterministic local optimiser. In most update steps it performs
    either one evaluation, or two sequential evaluations, so that it will not
    typically benefit from parallelisation.

    Note that this optimiser does not support boundary constraints.

    Parameters
    ----------
    problem: pybop.Problem
        the cost function to be minimised.
    options: pybop.PintsOptions
        optimisation options

    See Also
    --------
    pints.NelderMead : PINTS implementation of Nelder-Mead algorithm.
    """

    def __init__(
        self,
        problem: Problem,
        options: Optional[pybop.PintsOptions],
    ):
        super().__init__(
            problem,
            PintsNelderMead,
            options,
        )


class CMAES(BasePintsOptimiser):
    """
    Adapter for the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) optimiser in PINTS.

    CMA-ES is an evolutionary algorithm for difficult non-linear non-convex optimisation problems.
    It adapts the covariance matrix of a multivariate normal distribution to capture the shape of
    the cost landscape.

    Parameters
    ----------
    problem: pybop.Problem
        the cost function to be minimised.
    options: pybop.PintsOptions
        optimisation options

    See Also
    --------
    pints.CMAES : PINTS implementation of CMA-ES algorithm.
    """

    def __init__(
        self,
        problem: Problem,
        options: Optional[pybop.PintsOptions],
    ):
        if len(problem.params) == 1:
            raise ValueError(
                "CMAES requires optimisation of >= 2 parameters at once. "
                "Please choose another optimiser."
            )
        super().__init__(
            problem,
            PintsCMAES,
            options,
        )


class CuckooSearch(BasePintsOptimiser):
    """
    Adapter for the Cuckoo Search optimiser in PyBOP.

    Cuckoo Search is a population-based optimisation algorithm inspired by the brood parasitism of some cuckoo species.
    It is designed to be simple, efficient, and robust, and is suitable for global optimisation problems.

    Parameters
    ----------
    problem: pybop.Problem
        the cost function to be minimised.
    options: pybop.PintsOptions
        optimisation options

    See Also
    --------
    pybop.CuckooSearchImpl : PyBOP implementation of Cuckoo Search algorithm.
    """

    def __init__(
        self,
        problem: Problem,
        options: Optional[pybop.PintsOptions],
    ):
        super().__init__(
            problem,
            CuckooSearchImpl,
            options,
        )


class RandomSearch(BasePintsOptimiser):
    """
    Adapter for the Random Search optimiser in PyBOP.

    Random Search is a simple optimisation algorithm that samples parameter sets randomly
    within the given boundaries and identifies the best solution based on fitness.

    This optimiser has been implemented for benchmarking and comparisons, convergence will be
    better with one of other optimisers in the majority of cases.

    Parameters
    ----------
    problem: pybop.Problem
        the cost function to be minimised.
    options: pybop.PintsOptions
        optimisation options

    See Also
    --------
    pybop.RandomSearchImpl : PyBOP implementation of Random Search algorithm.
    """

    def __init__(
        self,
        problem: Problem,
        options: Optional[pybop.PintsOptions],
    ):
        super().__init__(
            problem,
            RandomSearchImpl,
            options,
        )


class SimulatedAnnealing(BasePintsOptimiser):
    """
    Adapter for Simulated Annealing optimiser in PyBOP.

    Simulated Annealing is a probabilistic optimisation algorithm inspired by the annealing
    process in metallurgy. It works by iteratively proposing new solutions and accepting
    them based on both their fitness and a temperature parameter that decreases over time.
    This allows the algorithm to initially explore broadly and gradually focus on local
    optimisation as the temperature decreases.

    The algorithm is particularly effective at avoiding local minima and returning a
    global solution.

    Parameters
    ----------
    problem: pybop.Problem
        the cost function to be minimised.
    options: pybop.PintsOptions
        optimisation options

    See Also
    --------
    pybop.SimulatedAnnealingImpl : PyBOP implementation of Simulated Annealing algorithm.
    """

    def __init__(
        self,
        problem: Problem,
        options: Optional[pybop.PintsOptions],
    ):
        super().__init__(
            problem,
            SimulatedAnnealingImpl,
            options,
        )
