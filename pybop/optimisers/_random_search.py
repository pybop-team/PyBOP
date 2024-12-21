import numpy as np
from pints import PopulationBasedOptimiser


class RandomSearchImpl(PopulationBasedOptimiser):
    """
    Random Search (RS) optimisation algorithm.
    This algorithm explores the parameter space by randomly sampling points.

    The algorithm does the following:
    1. Initialise a population of solutions.
    2. At each iteration, generate `n` number of random positions within boundaries.
    3. Evaluate the quality/fitness of the positions.
    4. Replace the best position with improved position if found.

    Parameters:
        population_size (optional): Number of solutions to evaluate per iteration.

    References:
    The Random Search algorithm implemented in this work is based on principles outlined
    in "Introduction to Stochastic Search and Optimization: Estimation, Simulation, and
    Control" by Spall, J. C. (2003).

    The implementation inherits from the PINTS PopulationOptimiser.
    """

    def __init__(self, x0, sigma0=0.05, boundaries=None):
        super().__init__(x0, sigma0, boundaries=boundaries)

        # Problem dimensionality
        self._dim = len(x0)

        # Initialise best solution
        self._x_best = np.copy(x0)
        self._f_best = np.inf
        self._running = False
        self._ready_for_tell = False

    def ask(self):
        """
        Returns a list of positions to evaluate in the optimiser-space.
        """
        self._ready_for_tell = True
        self._running = True

        # Generate random solutions
        if self._boundaries:
            self._candidates = np.random.uniform(
                low=self._boundaries.lower(),
                high=self._boundaries.upper(),
                size=(self._population_size, self._dim),
            )
            return self._candidates

        self._candidates = np.random.normal(
            self._x0, self._sigma0, size=(self._population_size, self._dim)
        )
        return self.clip_candidates(self._candidates)

    def tell(self, replies):
        """
        Receives a list of cost function values from points previously specified
        by `self.ask()`, and updates the optimiser state accordingly.
        """
        if not self._ready_for_tell:
            raise RuntimeError("ask() must be called before tell().")

        # Evaluate solutions and update the best
        for i in range(self._population_size):
            f_new = replies[i]
            if f_new < self._f_best:
                self._f_best = f_new
                self._x_best = self._candidates[i]

    def running(self):
        """
        Returns ``True`` if the optimisation is in progress.
        """
        return self._running

    def x_best(self):
        """
        Returns the best parameter values found so far.
        """
        return self._x_best

    def f_best(self):
        """
        Returns the best score found so far.
        """
        return self._f_best

    def name(self):
        """
        Returns the name of the optimiser.
        """
        return "Random Search"

    def clip_candidates(self, x):
        """
        Clip the input array to the boundaries if available.
        """
        if self._boundaries:
            x = np.clip(x, self._boundaries.lower(), self._boundaries.upper())
        return x

    def _suggested_population_size(self):
        """
        Returns a suggested population size based on the dimension of the parameter space.
        """
        return 4 + int(3 * np.log(self._n_parameters))
