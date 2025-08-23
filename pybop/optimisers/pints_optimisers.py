from pints import CMAES as PintsCMAES
from pints import PSO as PintsPSO
from pints import SNES as PintsSNES
from pints import XNES as PintsXNES
from pints import IRPropMin as PintsIRPropMin
from pints import NelderMead as PintsNelderMead

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
from pybop.problems.base_problem import Problem


class GradientDescent(BasePintsOptimiser):
    """
    Implements a simple gradient descent optimisation algorithm. Gradient descent is a canonical
    method selecting a proposal from the previous proposal alongside the corresponding cost
    gradient wrt. the parameters. Due to the fixed step-size convergence rate commonly decreases
    as the gradient shrinks when approaching a local minima.

    This class extends the gradient descent optimiser from the PINTS library, designed to minimise
    a scalar function of one or more variables.

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
        options: pybop.PintsOptions | None = None,
    ):
        super().__init__(
            problem,
            GradientDescentImpl,
            options,
        )


class AdamW(BasePintsOptimiser):
    """
    Implements the Adaptive Moment Estimation with Weight Decay (AdamW) optimisation algorithm.

    This class extends the AdamW optimiser, which is a variant of the Adam optimiser that
    incorporates weight decay. AdamW is designed to be more robust and stable for training deep
    neural networks, particularly when using larger learning rates.

    Note: This optimiser does not support boundary constraints.

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
        options: pybop.PintsOptions | None = None,
    ):
        super().__init__(
            problem,
            AdamWImpl,
            options,
        )


class IRPropMin(BasePintsOptimiser):
    """
    Implements the iRpropMin optimisation algorithm. This method uses gradient information for the
    proposal direction with a separated step-size.

    This class inherits from the PINTS IRPropMin class, which is an optimiser that uses resilient
    backpropagation without weight-backtracking. It is designed to handle problems with large
    plateaus, noisy gradients, and local minima.

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
        options: pybop.PintsOptions | None = None,
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
        options: pybop.PintsOptions | None = None,
    ):
        super().__init__(
            problem,
            IRPropPlusImpl,
            options,
        )


class PSO(BasePintsOptimiser):
    """
    Implements a particle swarm optimisation (PSO) algorithm. This method is a heuristic
    population based method which aims to emulate the dynamics of natural phenomena. This is
    implemented as "particles" moving around the search space. Global optima convergence is
    guaranteed in the infinite limit for the number of optimiser iterations.

    This class extends the PSO optimiser from the PINTS library. PSO is a metaheuristic
    optimisation method inspired by the social behavior of birds flocking or fish schooling,
    suitable for global optimisation problems.

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
        options: pybop.PintsOptions | None = None,
    ):
        super().__init__(
            problem,
            PintsPSO,
            options,
        )


class SNES(BasePintsOptimiser):
    """
    Implements the stochastic natural evolution strategy (SNES) optimisation algorithm. SNES is a
    population-based heuristic algorithm which constructs proposals based on replicating the
    natural gradient through evolution based on previous evaluations.

    Inheriting from the PINTS SNES class, this optimiser is an evolutionary algorithm that evolves
    a probability distribution on the parameter space, guiding the search for the optimum based on
    the natural gradient of expected fitness.

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
        options: pybop.PintsOptions | None = None,
    ):
        super().__init__(
            problem,
            PintsSNES,
            options,
        )


class XNES(BasePintsOptimiser):
    """
    Implements the Exponential Natural Evolution Strategy (XNES) optimiser from PINTS.
    XNES is a population-based heuristic algorithm which samples from a normal distribution for
    candidate proposals while updating this distribution from the cost landscape.

    XNES is an evolutionary algorithm that samples from a multivariate normal distribution, which
    is updated iteratively to fit the distribution of successful solutions.

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
        options: pybop.PintsOptions | None = None,
    ):
        super().__init__(
            problem,
            PintsXNES,
            options,
        )


class NelderMead(BasePintsOptimiser):
    """
    Implements the Nelder-Mead downhill simplex method from PINTS. Alternatively, known as the
    downhill simplex method, it's a heuristic method that does not use gradient information.
    Nelder-Mead is a conventionally robust implementation for optimisation in electrochemical
    problems.

    This is a deterministic local optimiser. In most update steps it performs either one
    evaluation, or two sequential evaluations, so that it will not typically benefit from
    parallelisation.

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
        options: pybop.PintsOptions | None = None,
    ):
        super().__init__(
            problem,
            PintsNelderMead,
            options,
        )


class CMAES(BasePintsOptimiser):
    """
    Adapter for the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) optimiser in PINTS.
    This is a population-based heuristic method that doesn't use gradient information. CMA-ES is
    quite robust for general identification / optimisation tasks.

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
        options: pybop.PintsOptions | None = None,
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
    Adapter for the Cuckoo Search optimiser in PyBOP. Cuckoo is a population-based algorithm which
    explores the search space by randomly suggesting candidates "nests" and  abandoning poorly
    performing "nests" throughout the process.

    Cuckoo Search is a population-based optimisation algorithm inspired by the brood parasitism of
    some cuckoo species. It is designed to be simple, efficient, and robust, and is suitable for
    global optimisation problems.

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
        options: pybop.PintsOptions | None = None,
    ):
        super().__init__(
            problem,
            CuckooSearchImpl,
            options,
        )


class RandomSearch(BasePintsOptimiser):
    """
    Adapter for the Random Search optimiser in PyBOP. Random search is a helpful benchmark to
    compare new optimiser implementations. Random search samples from the parameter space
    uniformly and stores the current best proposal assessed from the cost landscape.

    Random Search is a simple optimisation algorithm that samples parameter sets randomly within
    the given boundaries and identifies the best solution based on fitness.

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
        options: pybop.PintsOptions | None = None,
    ):
        super().__init__(
            problem,
            RandomSearchImpl,
            options,
        )


class SimulatedAnnealing(BasePintsOptimiser):
    """
    Adapter for Simulated Annealing optimiser in PyBOP. Simulated Annealing is a probabilistic
    optimisation method inspired by the annealing process in metallurgy. It works by iteratively
    proposing new solutions and accepting them based on both their fitness and a temperature
    parameter that decreases over time.  This allows the algorithm to initially explore broadly
    and gradually focus on local optimisation as the temperature decreases.

    Simulated Annealing is a probabilistic optimisation algorithm inspired by the annealing
    process in metallurgy. It works by iteratively proposing new solutions and accepting them
    based on both their fitness and a temperature parameter that decreases over time. This allows
    the algorithm to initially explore broadly and gradually focus on local optimisation as the
    temperature decreases.

    The algorithm is particularly effective at avoiding local minima and returning a global
    solution.

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
        options: pybop.PintsOptions | None = None,
    ):
        super().__init__(
            problem,
            SimulatedAnnealingImpl,
            options,
        )
