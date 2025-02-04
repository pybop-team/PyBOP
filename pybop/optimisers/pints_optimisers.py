from pints import CMAES as PintsCMAES
from pints import PSO as PintsPSO
from pints import SNES as PintsSNES
from pints import XNES as PintsXNES
from pints import IRPropMin as PintsIRPropMin
from pints import NelderMead as PintsNelderMead

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
    cost : callable
        The cost function to be minimised.
    max_iterations : int, optional
        Maximum number of iterations for the optimisation.
    min_iterations : int, optional (default=2)
        Minimum number of iterations before termination.
    max_unchanged_iterations : int, optional (default=15)
        Maximum number of iterations without improvement before termination.
    multistart : int, optional (default=1)
        Number of optimiser restarts from randomly sample position. These positions
        are sampled from the priors.
    parallel : bool, optional (default=False)
        Whether to run the optimisation in parallel.
    **optimiser_kwargs : optional
        Valid PINTS option keys and their values, for example:
        x0 : array_like
            Initial position from which optimisation will start.
        sigma0 : float
            Initial step size or standard deviation depending on the optimiser.
        bounds : dict
            A dictionary with 'lower' and 'upper' keys containing arrays for lower and
            upper bounds on the parameters.
        use_f_guessed : bool
            Whether to return the guessed function values.
        absolute_tolerance : float
            Absolute tolerance for convergence checking.
        relative_tolerance : float
            Relative tolerance for convergence checking.
        max_evaluations : int
            Maximum number of function evaluations.
        threshold : float
            Threshold value for early termination.

    See Also
    --------
    pints.GradientDescent : The PINTS implementation this class is based on.
    """

    def __init__(
        self,
        cost,
        max_iterations: int = None,
        min_iterations: int = 2,
        max_unchanged_iterations: int = 15,
        multistart: int = 1,
        parallel: bool = False,
        **optimiser_kwargs,
    ):
        super().__init__(
            cost,
            GradientDescentImpl,
            max_iterations,
            min_iterations,
            max_unchanged_iterations,
            multistart,
            parallel,
            **optimiser_kwargs,
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
    cost : callable
        The cost function to be minimised.
    max_iterations : int, optional
        Maximum number of iterations for the optimisation.
    min_iterations : int, optional (default=2)
        Minimum number of iterations before termination.
    max_unchanged_iterations : int, optional (default=15)
        Maximum number of iterations without improvement before termination.
    multistart : int, optional (default=1)
        Number of optimiser restarts from randomly sample position. These positions
        are sampled from the priors.
    parallel : bool, optional (default=False)
        Whether to run the optimisation in parallel.
    **optimiser_kwargs : optional
        Valid PINTS option keys and their values, for example:
        x0 : array_like
            Initial position from which optimisation will start.
        sigma0 : float
            Initial step size or standard deviation depending on the optimiser.
        bounds : dict
            A dictionary with 'lower' and 'upper' keys containing arrays for lower and
            upper bounds on the parameters.
        use_f_guessed : bool
            Whether to return the guessed function values.
        absolute_tolerance : float
            Absolute tolerance for convergence checking.
        relative_tolerance : float
            Relative tolerance for convergence checking.
        max_evaluations : int
            Maximum number of function evaluations.
        threshold : float
            Threshold value for early termination.

    See Also
    --------
    pybop.AdamWImpl : The PyBOP implementation this class is based on.
    """

    def __init__(
        self,
        cost,
        max_iterations: int = None,
        min_iterations: int = 2,
        max_unchanged_iterations: int = 15,
        multistart: int = 1,
        parallel: bool = False,
        **optimiser_kwargs,
    ):
        super().__init__(
            cost,
            AdamWImpl,
            max_iterations,
            min_iterations,
            max_unchanged_iterations,
            multistart,
            parallel,
            **optimiser_kwargs,
        )


class IRPropMin(BasePintsOptimiser):
    """
    Implements the iRpropMin optimisation algorithm.

    This class inherits from the PINTS IRPropMin class, which is an optimiser that
    uses resilient backpropagation without weight-backtracking. It is designed to handle
    problems with large plateaus, noisy gradients, and local minima.

    Parameters
    ----------
    cost : callable
        The cost function to be minimised.
    max_iterations : int, optional
        Maximum number of iterations for the optimisation.
    min_iterations : int, optional (default=2)
        Minimum number of iterations before termination.
    max_unchanged_iterations : int, optional (default=15)
        Maximum number of iterations without improvement before termination.
    multistart : int, optional (default=1)
        Number of optimiser restarts from randomly sample position. These positions
        are sampled from the priors.
    parallel : bool, optional (default=False)
        Whether to run the optimisation in parallel.
    **optimiser_kwargs : optional
        Valid PINTS option keys and their values, for example:
        x0 : array_like
            Initial position from which optimisation will start.
        sigma0 : float
            Initial step size or standard deviation depending on the optimiser.
        bounds : dict
            A dictionary with 'lower' and 'upper' keys containing arrays for lower and
            upper bounds on the parameters.
        use_f_guessed : bool
            Whether to return the guessed function values.
        absolute_tolerance : float
            Absolute tolerance for convergence checking.
        relative_tolerance : float
            Relative tolerance for convergence checking.
        max_evaluations : int
            Maximum number of function evaluations.
        threshold : float
            Threshold value for early termination.

    See Also
    --------
    pints.IRPropMin : The PINTS implementation this class is based on.
    """

    def __init__(
        self,
        cost,
        max_iterations: int = None,
        min_iterations: int = 2,
        max_unchanged_iterations: int = 15,
        multistart: int = 1,
        parallel: bool = False,
        **optimiser_kwargs,
    ):
        super().__init__(
            cost,
            PintsIRPropMin,
            max_iterations,
            min_iterations,
            max_unchanged_iterations,
            multistart,
            parallel,
            **optimiser_kwargs,
        )


class IRPropPlus(BasePintsOptimiser):
    """
    Implements the iRpropPlus optimisation algorithm.

    This class implements the improved resilient backpropagation with weight-backtracking.
    It is designed to handle problems with large plateaus, noisy gradients, and local minima.

    Parameters
    ----------
    cost : callable
        The cost function to be minimized.
    max_iterations : int, optional
        Maximum number of iterations for the optimisation.
    min_iterations : int, optional (default=2)
        Minimum number of iterations before termination.
    max_unchanged_iterations : int, optional (default=15)
        Maximum number of iterations without improvement before termination.
    multistart : int, optional (default=1)
        Number of optimiser restarts from randomly sample position. These positions
        are sampled from the priors.
    parallel : bool, optional (default=False)
        Whether to run the optimisation in parallel.
    **optimiser_kwargs : optional
        Valid PINTS option keys and their values, for example:
        x0 : array_like
            Initial position from which optimisation will start.
        sigma0 : float
            Initial step size or standard deviation depending on the optimiser.
        bounds : dict
            A dictionary with 'lower' and 'upper' keys containing arrays for lower and
            upper bounds on the parameters.
        use_f_guessed : bool
            Whether to return the guessed function values.
        absolute_tolerance : float
            Absolute tolerance for convergence checking.
        relative_tolerance : float
            Relative tolerance for convergence checking.
        max_evaluations : int
            Maximum number of function evaluations.
        threshold : float
            Threshold value for early termination.

    See Also
    --------
    pints.IRPropMin : The PINTS implementation this class is based on.
    """

    def __init__(
        self,
        cost,
        max_iterations: int = None,
        min_iterations: int = 2,
        max_unchanged_iterations: int = 15,
        multistart: int = 1,
        parallel: bool = False,
        **optimiser_kwargs,
    ):
        super().__init__(
            cost,
            IRPropPlusImpl,
            max_iterations,
            min_iterations,
            max_unchanged_iterations,
            multistart,
            parallel,
            **optimiser_kwargs,
        )


class PSO(BasePintsOptimiser):
    """
    Implements a particle swarm optimisation (PSO) algorithm.

    This class extends the PSO optimiser from the PINTS library. PSO is a
    metaheuristic optimisation method inspired by the social behavior of birds
    flocking or fish schooling, suitable for global optimisation problems.

    Parameters
    ----------
    cost : callable
        The cost function to be minimised.
    max_iterations : int, optional
        Maximum number of iterations for the optimisation.
    min_iterations : int, optional (default=2)
        Minimum number of iterations before termination.
    max_unchanged_iterations : int, optional (default=15)
        Maximum number of iterations without improvement before termination.
    multistart : int, optional (default=1)
        Number of optimiser restarts from randomly sample position. These positions
        are sampled from the priors.
    parallel : bool, optional (default=False)
        Whether to run the optimisation in parallel.
    **optimiser_kwargs : optional
        Valid PINTS option keys and their values, for example:
        x0 : array_like
            Initial position from which optimisation will start.
        sigma0 : float
            Initial step size or standard deviation depending on the optimiser.
        bounds : dict
            A dictionary with 'lower' and 'upper' keys containing arrays for lower and
            upper bounds on the parameters.
        use_f_guessed : bool
            Whether to return the guessed function values.
        absolute_tolerance : float
            Absolute tolerance for convergence checking.
        relative_tolerance : float
            Relative tolerance for convergence checking.
        max_evaluations : int
            Maximum number of function evaluations.
        threshold : float
            Threshold value for early termination.

    See Also
    --------
    pints.PSO : The PINTS implementation this class is based on.
    """

    def __init__(
        self,
        cost,
        max_iterations: int = None,
        min_iterations: int = 2,
        max_unchanged_iterations: int = 15,
        multistart: int = 1,
        parallel: bool = False,
        **optimiser_kwargs,
    ):
        super().__init__(
            cost,
            PintsPSO,
            max_iterations,
            min_iterations,
            max_unchanged_iterations,
            multistart,
            parallel,
            **optimiser_kwargs,
        )


class SNES(BasePintsOptimiser):
    """
    Implements the stochastic natural evolution strategy (SNES) optimisation algorithm.

    Inheriting from the PINTS SNES class, this optimiser is an evolutionary algorithm
    that evolves a probability distribution on the parameter space, guiding the search
    for the optimum based on the natural gradient of expected fitness.

    Parameters
    ----------
    cost : callable
        The cost function to be minimised.
    max_iterations : int, optional
        Maximum number of iterations for the optimisation.
    min_iterations : int, optional (default=2)
        Minimum number of iterations before termination.
    max_unchanged_iterations : int, optional (default=15)
        Maximum number of iterations without improvement before termination.
    multistart : int, optional (default=1)
        Number of optimiser restarts from randomly sample position. These positions
        are sampled from the priors.
    parallel : bool, optional (default=False)
        Whether to run the optimisation in parallel.
    **optimiser_kwargs : optional
        Valid PINTS option keys and their values, for example:
        x0 : array_like
            Initial position from which optimisation will start.
        sigma0 : float
            Initial step size or standard deviation depending on the optimiser.
        bounds : dict
            A dictionary with 'lower' and 'upper' keys containing arrays for lower and
            upper bounds on the parameters.
        use_f_guessed : bool
            Whether to return the guessed function values.
        absolute_tolerance : float
            Absolute tolerance for convergence checking.
        relative_tolerance : float
            Relative tolerance for convergence checking.
        max_evaluations : int
            Maximum number of function evaluations.
        threshold : float
            Threshold value for early termination.

    See Also
    --------
    pints.SNES : The PINTS implementation this class is based on.
    """

    def __init__(
        self,
        cost,
        max_iterations: int = None,
        min_iterations: int = 2,
        max_unchanged_iterations: int = 15,
        multistart: int = 1,
        parallel: bool = False,
        **optimiser_kwargs,
    ):
        super().__init__(
            cost,
            PintsSNES,
            max_iterations,
            min_iterations,
            max_unchanged_iterations,
            multistart,
            parallel,
            **optimiser_kwargs,
        )


class XNES(BasePintsOptimiser):
    """
    Implements the Exponential Natural Evolution Strategy (XNES) optimiser from PINTS.

    XNES is an evolutionary algorithm that samples from a multivariate normal
    distribution, which is updated iteratively to fit the distribution of successful
    solutions.

    Parameters
    ----------
    cost : callable
        The cost function to be minimised.
    max_iterations : int, optional
        Maximum number of iterations for the optimisation.
    min_iterations : int, optional (default=2)
        Minimum number of iterations before termination.
    max_unchanged_iterations : int, optional (default=15)
        Maximum number of iterations without improvement before termination.
    multistart : int, optional (default=1)
        Number of optimiser restarts from randomly sample position. These positions
        are sampled from the priors.
    parallel : bool, optional (default=False)
        Whether to run the optimisation in parallel.
    **optimiser_kwargs : optional
        Valid PINTS option keys and their values, for example:
        x0 : array_like
            Initial position from which optimisation will start.
        sigma0 : float
            Initial step size or standard deviation depending on the optimiser.
        bounds : dict
            A dictionary with 'lower' and 'upper' keys containing arrays for lower and
            upper bounds on the parameters.
        use_f_guessed : bool
            Whether to return the guessed function values.
        absolute_tolerance : float
            Absolute tolerance for convergence checking.
        relative_tolerance : float
            Relative tolerance for convergence checking.
        max_evaluations : int
            Maximum number of function evaluations.
        threshold : float
            Threshold value for early termination.

    See Also
    --------
    pints.XNES : PINTS implementation of XNES algorithm.
    """

    def __init__(
        self,
        cost,
        max_iterations: int = None,
        min_iterations: int = 2,
        max_unchanged_iterations: int = 15,
        multistart: int = 1,
        parallel: bool = False,
        **optimiser_kwargs,
    ):
        super().__init__(
            cost,
            PintsXNES,
            max_iterations,
            min_iterations,
            max_unchanged_iterations,
            multistart,
            parallel,
            **optimiser_kwargs,
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
    cost : callable
        The cost function to be minimised.
    max_iterations : int, optional
        Maximum number of iterations for the optimisation.
    min_iterations : int, optional (default=2)
        Minimum number of iterations before termination.
    max_unchanged_iterations : int, optional (default=15)
        Maximum number of iterations without improvement before termination.
    multistart : int, optional (default=1)
        Number of optimiser restarts from randomly sample position. These positions
        are sampled from the priors.
    parallel : bool, optional (default=False)
        Whether to run the optimisation in parallel.
    **optimiser_kwargs : optional
        Valid PINTS option keys and their values, for example:
        x0 : array_like
            Initial position from which optimisation will start.
        sigma0 : float
            Initial step size or standard deviation depending on the optimiser.
        bounds : dict
            A dictionary with 'lower' and 'upper' keys containing arrays for lower and
            upper bounds on the parameters.
        use_f_guessed : bool
            Whether to return the guessed function values.
        absolute_tolerance : float
            Absolute tolerance for convergence checking.
        relative_tolerance : float
            Relative tolerance for convergence checking.
        max_evaluations : int
            Maximum number of function evaluations.
        threshold : float
            Threshold value for early termination.

    See Also
    --------
    pints.NelderMead : PINTS implementation of Nelder-Mead algorithm.
    """

    def __init__(
        self,
        cost,
        max_iterations: int = None,
        min_iterations: int = 2,
        max_unchanged_iterations: int = 15,
        multistart: int = 1,
        parallel: bool = False,
        **optimiser_kwargs,
    ):
        super().__init__(
            cost,
            PintsNelderMead,
            max_iterations,
            min_iterations,
            max_unchanged_iterations,
            multistart,
            parallel,
            **optimiser_kwargs,
        )


class CMAES(BasePintsOptimiser):
    """
    Adapter for the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) optimiser in PINTS.

    CMA-ES is an evolutionary algorithm for difficult non-linear non-convex optimisation problems.
    It adapts the covariance matrix of a multivariate normal distribution to capture the shape of
    the cost landscape.

    Parameters
    ----------
    cost : callable
        The cost function to be minimised.
    max_iterations : int, optional
        Maximum number of iterations for the optimisation.
    min_iterations : int, optional (default=2)
        Minimum number of iterations before termination.
    max_unchanged_iterations : int, optional (default=15)
        Maximum number of iterations without improvement before termination.
    multistart : int, optional (default=1)
        Number of optimiser restarts from randomly sample position. These positions
        are sampled from the priors.
    parallel : bool, optional (default=False)
        Whether to run the optimisation in parallel.
    **optimiser_kwargs : optional
        Valid PINTS option keys and their values, for example:
        x0 : array_like
            Initial position from which optimisation will start.
        sigma0 : float
            Initial step size or standard deviation depending on the optimiser.
        bounds : dict
            A dictionary with 'lower' and 'upper' keys containing arrays for lower and
            upper bounds on the parameters.
        use_f_guessed : bool
            Whether to return the guessed function values.
        absolute_tolerance : float
            Absolute tolerance for convergence checking.
        relative_tolerance : float
            Relative tolerance for convergence checking.
        max_evaluations : int
            Maximum number of function evaluations.
        threshold : float
            Threshold value for early termination.

    See Also
    --------
    pints.CMAES : PINTS implementation of CMA-ES algorithm.
    """

    def __init__(
        self,
        cost,
        max_iterations: int = None,
        min_iterations: int = 2,
        max_unchanged_iterations: int = 15,
        multistart: int = 1,
        parallel: bool = False,
        **optimiser_kwargs,
    ):
        x0 = optimiser_kwargs.get("x0", cost.parameters.initial_value())
        if len(x0) == 1 or len(cost.parameters) == 1:
            raise ValueError(
                "CMAES requires optimisation of >= 2 parameters at once. "
                "Please choose another optimiser."
            )
        super().__init__(
            cost,
            PintsCMAES,
            max_iterations,
            min_iterations,
            max_unchanged_iterations,
            multistart,
            parallel,
            **optimiser_kwargs,
        )


class CuckooSearch(BasePintsOptimiser):
    """
    Adapter for the Cuckoo Search optimiser in PyBOP.

    Cuckoo Search is a population-based optimisation algorithm inspired by the brood parasitism of some cuckoo species.
    It is designed to be simple, efficient, and robust, and is suitable for global optimisation problems.

    Parameters
    ----------
    cost : callable
        The cost function to be minimised.
    max_iterations : int, optional
        Maximum number of iterations for the optimisation.
    min_iterations : int, optional (default=2)
        Minimum number of iterations before termination.
    max_unchanged_iterations : int, optional (default=15)
        Maximum number of iterations without improvement before termination.
    multistart : int, optional (default=1)
        Number of optimiser restarts from randomly sample position. These positions
        are sampled from the priors.
    parallel : bool, optional (default=False)
        Whether to run the optimisation in parallel.
    **optimiser_kwargs : optional
        Valid PINTS option keys and their values, for example:
        x0 : array_like
            Initial position from which optimisation will start.
        sigma0 : float
            Initial step size or standard deviation depending on the optimiser.
        bounds : dict
            A dictionary with 'lower' and 'upper' keys containing arrays for lower and
            upper bounds on the parameters.
        use_f_guessed : bool
            Whether to return the guessed function values.
        absolute_tolerance : float
            Absolute tolerance for convergence checking.
        relative_tolerance : float
            Relative tolerance for convergence checking.
        max_evaluations : int
            Maximum number of function evaluations.
        threshold : float
            Threshold value for early termination.

    See Also
    --------
    pybop.CuckooSearchImpl : PyBOP implementation of Cuckoo Search algorithm.
    """

    def __init__(
        self,
        cost,
        max_iterations: int = None,
        min_iterations: int = 2,
        max_unchanged_iterations: int = 15,
        multistart: int = 1,
        parallel: bool = False,
        **optimiser_kwargs,
    ):
        super().__init__(
            cost,
            CuckooSearchImpl,
            max_iterations,
            min_iterations,
            max_unchanged_iterations,
            multistart,
            parallel,
            **optimiser_kwargs,
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
    cost : callable
        The cost function to be minimised.
    max_iterations : int, optional
        Maximum number of iterations for the optimisation.
    min_iterations : int, optional (default=2)
        Minimum number of iterations before termination.
    max_unchanged_iterations : int, optional (default=15)
        Maximum number of iterations without improvement before termination.
    multistart : int, optional (default=1)
        Number of optimiser restarts from randomly sample position. These positions
        are sampled from the priors.
    parallel : bool, optional (default=False)
        Whether to run the optimisation in parallel.
    **optimiser_kwargs : optional
        Valid PINTS option keys and their values, for example:
        x0 : array_like
            Initial position from which optimisation will start.
        population_size : int
            Number of solutions to evaluate per iteration.
        bounds : dict
            A dictionary with 'lower' and 'upper' keys containing arrays for lower and
            upper bounds on the parameters.
        absolute_tolerance : float
            Absolute tolerance for convergence checking.
        relative_tolerance : float
            Relative tolerance for convergence checking.
        max_evaluations : int
            Maximum number of function evaluations.
        threshold : float
            Threshold value for early termination.

    See Also
    --------
    pybop.RandomSearchImpl : PyBOP implementation of Random Search algorithm.
    """

    def __init__(
        self,
        cost,
        max_iterations: int = None,
        min_iterations: int = 2,
        max_unchanged_iterations: int = 15,
        multistart: int = 1,
        parallel: bool = False,
        **optimiser_kwargs,
    ):
        super().__init__(
            cost,
            RandomSearchImpl,
            max_iterations,
            min_iterations,
            max_unchanged_iterations,
            multistart,
            parallel,
            **optimiser_kwargs,
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
    cost : callable
        The cost function to be minimised.
    max_iterations : int, optional
        Maximum number of iterations for the optimisation.
    min_iterations : int, optional (default=2)
        Minimum number of iterations before termination.
    max_unchanged_iterations : int, optional (default=15)
        Maximum number of iterations without improvement before termination.
    multistart : int, optional (default=1)
        Number of optimiser restarts from randomly sample position. These positions
        are sampled from the priors.
    parallel : bool, optional (default=False)
        Whether to run the optimisation in parallel.
    **optimiser_kwargs : optional
        Valid PINTS option keys and their values, for example:
        x0 : array_like
            Initial position from which optimisation will start.
        sigma0 : float
            Initial step size or standard deviation for parameter perturbation.
        bounds : dict
            A dictionary with 'lower' and 'upper' keys containing arrays for lower and
            upper bounds on the parameters.
        cooling_schedule : callable, optional
            Function that determines how temperature decreases over time.
        initial_temperature : float, optional
            Starting temperature for the annealing process.
        absolute_tolerance : float
            Absolute tolerance for convergence checking.
        relative_tolerance : float
            Relative tolerance for convergence checking.
        max_evaluations : int
            Maximum number of function evaluations.
        threshold : float
            Threshold value for early termination.

    See Also
    --------
    pybop.SimulatedAnnealingImpl : PyBOP implementation of Simulated Annealing algorithm.
    """

    def __init__(
        self,
        cost,
        max_iterations: int = None,
        min_iterations: int = 2,
        max_unchanged_iterations: int = 15,
        multistart: int = 1,
        parallel: bool = False,
        **optimiser_kwargs,
    ):
        super().__init__(
            cost,
            SimulatedAnnealingImpl,
            max_iterations,
            min_iterations,
            max_unchanged_iterations,
            multistart,
            parallel,
            **optimiser_kwargs,
        )
