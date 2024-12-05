import numpy as np
from pints import PopulationBasedOptimiser


class RandomSearchImpl(PopulationBasedOptimiser):
    """
    Random Search (RS) optimisation algorithm.
    This algorithm explores the parameter space by randomly sampling points.

    The algorithm is simple:
    1. Initialise a population of solutions.
    2. At each iteration, generate new random solutions within boundaries.
    3. Evaluate the quality/fitness of the solutions.
    4. Update the best solution found so far.
    
    Parameters:
    - population_size (optional): Number of solutions to evaluate per iteration.

    References:
    - The Random Search algorithm implemented in this work is based on principles outlined
    in "Introduction to Stochastic Search and Optimization: Estimation, Simulation, and
    Control" by Spall, J. C. (2003).
    The implementation leverages the pints library framework, which provides tools for
    population-based optimization methods.
    """

    def __init__(self, x0, sigma0=0.05, boundaries=None, population_size=None):
        # Problem dimensionality
        self._dim = len(x0)  # Initialize _dim first

        super().__init__(x0, sigma0, boundaries=boundaries)

        # Population size, defaulting to a suggested value
        self._population_size = population_size or self._suggested_population_size()
        self.step_size = self._sigma0

        # Initialise best solutions
        self._x_best = np.copy(x0)
        self._f_best = np.inf

        # Iteration counter
        self._iterations = 0

        # Flags
        self._running = False
        self._ready_for_tell = False

    def ask(self):
        """
        Returns a list of next points in the parameter-space
        to evaluate from the optimiser.
        """
        self._ready_for_tell = True
        self._running = True

        # Generate random solutions within the boundaries
        self._candidates = np.random.uniform(
            low=self._boundaries.lower(),
            high=self._boundaries.upper(),
            size=(self._population_size, self._dim),
        )
        return self._candidates

    def tell(self, replies):
        """
        Receives a list of function values from the cost function from points
        previously specified by `self.ask()`, and updates the optimiser state
        accordingly.
        """
        if not self._ready_for_tell:
            raise RuntimeError("Optimiser not ready for tell()")

        self._iterations += 1
        self._ready_for_tell = False

        # Update the best solution
        for i, fitness in enumerate(replies):
            if fitness < self._f_best:
                self._f_best = fitness
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

    def _suggested_population_size(self):
        """
        Returns a suggested population size based on the dimension of the parameter space.
        """
        return 10 + int(2 * np.log(self._dim))
