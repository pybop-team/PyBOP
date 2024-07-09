import numpy as np
from pints import PopulationBasedOptimiser
from scipy.special import gamma


class CuckooSearchImpl(PopulationBasedOptimiser):
    """
    Cuckoo Search (CS) optimisation algorithm, inspired by the brood parasitism
    of some cuckoo species. This algorithm was introduced by Yang and Deb in 2009.

    The algorithm uses a population of host nests (solutions), where each cuckoo
    (new solution) tries to replace a worse nest in the population. The quality
    or fitness of the nests is determined by the cost function. A fraction
    of the worst nests is abandoned at each generation, and new ones are built
    randomly.

    The pseudo-code for the Cuckoo Search is as follows:

    1. Initialise population of n host nests
    2. While (t < max_generations):
        a. Get a cuckoo randomly by Lévy flights
        b. Evaluate its quality/fitness F
        c. Choose a nest among n (say, j) randomly
        d. If (F > fitness of j):
            i. Replace j with the new solution
        e. Abandon a fraction (pa) of the worst nests and build new ones
        f. Keep the best solutions/nests
        g. Rank the solutions and find the current best
    3. End While

    This implementation also uses a decreasing step size for the Lévy flights, calculated
    as sigma = sigma0 / sqrt(iterations), where sigma0 is the initial step size and
    iterations is the current iteration number.

    Parameters:
    - pa: Probability of discovering alien eggs/solutions (abandoning rate)

    References:
    - X. -S. Yang and Suash Deb, "Cuckoo Search via Lévy flights,"
      2009 World Congress on Nature & Biologically Inspired Computing (NaBIC),
      Coimbatore, India, 2009, pp. 210-214, https://doi.org/10.1109/NABIC.2009.5393690.

    - S. Walton, O. Hassan, K. Morgan, M.R. Brown,
      Modified cuckoo search: A new gradient free optimisation algorithm,
      Chaos, Solitons & Fractals, Volume 44, Issue 9, 2011,
      Pages 710-718, ISSN 0960-0779,
      https://doi.org/10.1016/j.chaos.2011.06.004.
    """

    def __init__(self, x0, sigma0=0.05, boundaries=None, pa=0.25):
        super().__init__(x0, sigma0, boundaries=boundaries)

        # Problem dimensionality
        self._dim = len(x0)

        # Population size and abandon rate
        self._n = self._population_size
        self._pa = pa
        self.step_size = self._sigma0
        self.beta = 1.5

        # Set states
        self._running = False
        self._ready_for_tell = False

        # Initialise nests
        if self._boundaries:
            self._nests = np.random.uniform(
                low=self._boundaries.lower(),
                high=self._boundaries.upper(),
                size=(self._n, self._dim),
            )
        else:
            self._nests = np.random.normal(
                self._x0, self._sigma0, size=(self._n, self._dim)
            )

        self._fitness = np.full(self._n, np.inf)

        # Initialise best solutions
        self._x_best = np.copy(x0)
        self._f_best = np.inf

        # Set iteration count
        self._iterations = 0

    def ask(self):
        """
        Returns a list of next points in the parameter-space
        to evaluate from the optimiser.
        """
        # Set flag to indicate that the optimiser is ready to receive replies
        self._ready_for_tell = True
        self._running = True

        # Generate new solutions (cuckoos) by Lévy flights
        self.step_size = self._sigma0 / max(1, np.sqrt(self._iterations))
        step = self.levy_flight(self.beta, self._dim) * self.step_size
        self.cuckoos = self._nests + step
        return self.clip_nests(self.cuckoos)

    def tell(self, replies):
        """
        Receives a list of function values from the cost function from points
        previously specified by `self.ask()`, and updates the optimiser state
        accordingly.
        """
        # Update iteration count
        self._iterations += 1

        # Compare cuckoos with current nests
        for i in range(self._n):
            f_new = replies[i]
            if f_new < self._fitness[i]:
                self._nests[i] = self.cuckoos[i]
                self._fitness[i] = f_new
                if f_new < self._f_best:
                    self._f_best = f_new
                    self._x_best = self.cuckoos[i]

        # Abandon some worse nests
        n_abandon = int(self._pa * self._n)
        worst_nests = np.argsort(self._fitness)[-n_abandon:]
        for idx in worst_nests:
            self.abandon_nests(idx)
            self._fitness[idx] = np.inf  # reset fitness

    def levy_flight(self, alpha, size):
        """
        Generate step sizes via the Mantegna's algorithm for Levy flights
        """
        from numpy import pi, power, random, sin

        sigma_u = power(
            (gamma(1 + alpha) * sin(pi * alpha / 2))
            / (gamma((1 + alpha) / 2) * alpha * power(2, (alpha - 1) / 2)),
            1 / alpha,
        )
        sigma_v = 1

        u = random.normal(0, sigma_u, size=size)
        v = random.normal(0, sigma_v, size=size)
        step = u / power(abs(v), 1 / alpha)

        return step

    def abandon_nests(self, idx):
        """
        Updates the nests to abandon the worst performers and reinitialise.
        """
        if self._boundaries:
            self._nests[idx] = np.random.uniform(
                low=self._boundaries.lower(),
                high=self._boundaries.upper(),
            )
        else:
            self._nests[idx] = np.random.normal(self._x0, self._sigma0)

    def clip_nests(self, x):
        """
        Clip the input array to the boundaries if available.
        """
        if self._boundaries:
            x = np.clip(x, self._boundaries.lower(), self._boundaries.upper())
        return x

    def _suggested_population_size(self):
        """
        Inherited from Pints:PopulationBasedOptimiser.
        Returns a suggested population size, based on the
        dimension of the parameter space.
        """
        return 4 + int(3 * np.log(self._n_parameters))

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
        return "Cuckoo Search"
