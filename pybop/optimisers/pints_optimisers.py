from pints import CMAES as PintsCMAES
from pints import PSO as PintsPSO
from pints import SNES as PintsSNES
from pints import XNES as PintsXNES
from pints import IRPropMin as PintsIRPropMin
from pints import NelderMead as PintsNelderMead

import pybop
from pybop.optimisers._adamw import AdamWImpl
from pybop.optimisers._cuckoo import CuckooSearchImpl
from pybop.optimisers._gradient_descent import GradientDescentImpl
from pybop.optimisers._irprop_plus import IRPropPlusImpl
from pybop.optimisers._random_search import RandomSearchImpl
from pybop.optimisers._simulated_annealing import SimulatedAnnealingImpl
from pybop.optimisers.base_pints_optimiser import BasePintsOptimiser
from pybop.problems.base_problem import Problem

__all__: list[str] = [
    "GradientDescent",
    "AdamW",
    "IRPropMin",
    "IRPropPlus",
    "PSO",
    "SNES",
    "XNES",
    "NelderMead",
    "CMAES",
    "CuckooSearch",
    "RandomSearch",
    "SimulatedAnnealing",
]


class GradientDescent(BasePintsOptimiser):
    """
    Adapter for gradient descent, a canonical method that takes steps in the opposite direction
    of the cost gradient with respect to the parameters (does not support boundary constraints).

    Gradient descent is designed to minimise a scalar function of one or more variables. Due to
    the fixed step-size, the convergence rate commonly decreases as the gradient shrinks when
    approaching a local minima.

    Parameters
    ----------
    problem: pybop.Problem
        The problem to optimse.
    options: pybop.PintsOptions
        Optimisation options.

    See Also
    --------
    pints.GradientDescent : The PINTS implementation this class is based on.
    """

    def __init__(
        self,
        problem: pybop.Problem,
        options: pybop.PintsOptions | None = None,
    ):
        super().__init__(problem, GradientDescentImpl, options)


class AdamW(BasePintsOptimiser):
    """
    Adapter for adaptive moment estimation with weight decay (AdamW), a variant of the Adam
    optimiser which does not support boundary constraints.

    This optimiser is designed to be more robust and stable for training deep neural networks,
    particularly when using larger learning rates.

    Parameters
    ----------
    problem: pybop.Problem
        The problem to optimse.
    options: pybop.PintsOptions
        Optimisation options.

    See Also
    --------
    pybop.AdamWImpl : The PyBOP implementation this class is based on.
    """

    def __init__(
        self,
        problem: pybop.Problem,
        options: pybop.PintsOptions | None = None,
    ):
        super().__init__(problem, AdamWImpl, options)


class IRPropMin(BasePintsOptimiser):
    """
    Adapter for improved resilient backpropagation (without weight-backtracking), an optimisation
    algorithm designed to handle problems with large plateaus, noisy gradients, and local minima.

    This method uses gradient information for the proposal direction with a separated step-size.

    Parameters
    ----------
    problem: pybop.Problem
        The problem to optimse.
    options: pybop.PintsOptions
        Optimisation options.

    See Also
    --------
    pints.IRPropMin : The PINTS implementation this class is based on.
    """

    def __init__(
        self,
        problem: pybop.Problem,
        options: pybop.PintsOptions | None = None,
    ):
        super().__init__(problem, PintsIRPropMin, options)


class IRPropPlus(BasePintsOptimiser):
    """
    Adapter for improved resilient backpropagation with weight-backtracking, an optimisation
    algorithm designed to handle problems with large plateaus, noisy gradients, and local minima.

    Parameters
    ----------
    problem: pybop.Problem
        The problem to optimse.
    options: pybop.PintsOptions
        Optimisation options.

    See Also
    --------
    pints.IRPropMin : The PINTS implementation this class is based on.
    """

    def __init__(
        self,
        problem: pybop.Problem,
        options: pybop.PintsOptions | None = None,
    ):
        super().__init__(problem, IRPropPlusImpl, options)


class PSO(BasePintsOptimiser):
    """
    Adapter for particle swarm optimisation (PSO), a metaheuristic optimisation method inspired by
    the social behavior of birds flocking or fish schooling, suitable for global optimisation
    problems.

    The method considers "particles" moving around the search space. Global optima convergence is
    guaranteed in the infinite limit for the number of optimiser iterations.

    Parameters
    ----------
    problem: pybop.Problem
        The problem to optimse.
    options: pybop.PintsOptions
        Optimisation options.

    See Also
    --------
    pints.PSO : The PINTS implementation this class is based on.
    """

    def __init__(
        self,
        problem: pybop.Problem,
        options: pybop.PintsOptions | None = None,
    ):
        super().__init__(problem, PintsPSO, options)


class SNES(BasePintsOptimiser):
    """
    Adapter for the stochastic natural evolution strategy (SNES), an evolutionary algorithm that
    evolves a probability distribution on the parameter space, guiding the search for the optimum
    based on the natural gradient of expected fitness.

    Parameters
    ----------
    problem: pybop.Problem
        The problem to optimse.
    options: pybop.PintsOptions
        Optimisation options.

    See Also
    --------
    pints.SNES : The PINTS implementation this class is based on.
    """

    def __init__(
        self,
        problem: pybop.Problem,
        options: pybop.PintsOptions | None = None,
    ):
        super().__init__(problem, PintsSNES, options)


class XNES(BasePintsOptimiser):
    """
    Adapter for the exponential natural evolution strategy (XNES), an evolutionary algorithm that
    samples from a multivariate normal distribution, which is updated iteratively to fit the
    distribution of successful solutions.

    Parameters
    ----------
    problem: pybop.Problem
        The problem to optimse.
    options: pybop.PintsOptions
        Optimisation options.

    See Also
    --------
    pints.XNES : PINTS implementation of XNES algorithm.
    """

    def __init__(
        self,
        problem: pybop.Problem,
        options: pybop.PintsOptions | None = None,
    ):
        super().__init__(problem, PintsXNES, options)


class NelderMead(BasePintsOptimiser):
    """
    Adpater for the Nelder-Mead downhill simplex method, a deterministic local optimiser that does
    not use gradient information or support boundary constraints.

    In most update steps, it performs either one evaluation, or two sequential evaluations, so
    that it will not typically benefit from parallelisation.

    Parameters
    ----------
    problem: pybop.Problem
        The problem to optimse.
    options: pybop.PintsOptions
        Optimisation options.

    See Also
    --------
    pints.NelderMead : PINTS implementation of Nelder-Mead algorithm.
    """

    def __init__(
        self,
        problem: pybop.Problem,
        options: pybop.PintsOptions | None = None,
    ):
        super().__init__(problem, PintsNelderMead, options)


class CMAES(BasePintsOptimiser):
    """
    Adapter for the covariance matrix adaptation evolution strategy (CMA-ES), an evolutionary
    algorithm for difficult non-linear non-convex optimisation problems.

    It adapts the covariance matrix of a multivariate normal distribution to capture the shape of
    the cost landscape.

    Parameters
    ----------
    problem: pybop.Problem
        The problem to optimse.
    options: pybop.PintsOptions
        Optimisation options.

    See Also
    --------
    pints.CMAES : PINTS implementation of CMA-ES algorithm.
    """

    def __init__(
        self,
        problem: pybop.Problem,
        options: pybop.PintsOptions | None = None,
    ):
        if isinstance(problem, Problem) and len(problem.parameters) == 1:
            raise ValueError(
                "CMAES requires optimisation of >= 2 parameters at once. "
                "Please choose another optimiser."
            )
        super().__init__(problem, PintsCMAES, options)


class CuckooSearch(BasePintsOptimiser):
    """
    Adapter for cuckoo search, a population-based optimisation algorithm inspired by the brood
    parasitism of some cuckoo species which is suitable for global optimisation problems.

    Cuckoo search is designed to be simple, efficient, and robust. It explores the search space by
    randomly suggesting candidate "nests" and abandoning poorly performing "nests" throughout the
    process.

    Parameters
    ----------
    problem: pybop.Problem
        The problem to optimse.
    options: pybop.PintsOptions
        Optimisation options.

    See Also
    --------
    pybop.CuckooSearchImpl : PyBOP implementation of Cuckoo Search algorithm.
    """

    def __init__(
        self,
        problem: pybop.Problem,
        options: pybop.PintsOptions | None = None,
    ):
        super().__init__(problem, CuckooSearchImpl, options)


class RandomSearch(BasePintsOptimiser):
    """
    Adapter for random search, a simple algorithm which samples parameter values randomly and
    stores the current best proposal based on fitness (not recommended for optimisation).

    This optimiser has been implemented for benchmarking and comparisons, convergence will be
    better with one of other optimisers in the majority of cases.

    Parameters
    ----------
    problem: pybop.Problem
        The problem to optimse.
    options: pybop.PintsOptions
        Optimisation options.

    See Also
    --------
    pybop.RandomSearchImpl : PyBOP implementation of Random Search algorithm.
    """

    def __init__(
        self,
        problem: pybop.Problem,
        options: pybop.PintsOptions | None = None,
    ):
        super().__init__(problem, RandomSearchImpl, options)


class SimulatedAnnealing(BasePintsOptimiser):
    """
    Adapter for simulated annealing, a probabilistic optimisation method inspired by the annealing
    process in metallurgy which is suitable for global optimisation problems.

    It works by iteratively proposing new solutions and accepting them based on both their fitness
    and a temperature parameter that decreases over time. This allows the algorithm to initially
    explore broadly and gradually focus on local optimisation as the temperature decreases.

    Parameters
    ----------
    problem: pybop.Problem
        The problem to optimse.
    options: pybop.PintsOptions
        Optimisation options.

    See Also
    --------
    pybop.SimulatedAnnealingImpl : PyBOP implementation of Simulated Annealing algorithm.
    """

    def __init__(
        self,
        problem: pybop.Problem,
        options: pybop.PintsOptions | None = None,
    ):
        super().__init__(problem, SimulatedAnnealingImpl, options)
