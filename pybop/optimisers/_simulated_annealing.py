import numpy as np
from pints import Optimiser as PintsOptimiser


class SimulatedAnnealingImpl(PintsOptimiser):
    """
    Simulated Annealing optimiser, implementing the classic temperature-based
    probabilistic optimisation method.

    This method uses a temperature schedule to control the probability of accepting
    worse solutions as it explores the parameter space. As the temperature decreases,
    the algorithm becomes more selective, eventually converging to a local or global
    optimum.

    The probability of accepting a worse solution is given by::

    .. math::

        P(accept) = exp(-(f_{\text{new}} - fold)/T)

    The temperature decreases according to the cooling schedule::

    .. math::

        T = T0 * \alpha^k

    where:
    - :math: T0 is the initial temperature
    - :math: \alpha is the cooling rate (between 0 and 1)
    - :math: k is the iteration number

    Parameters
    ----------
    x0 : numpy array
        Initial position
    sigma0 : float
        Initial step size
    boundaries : dict, optional
        Optional boundaries for parameters
    """

    def __init__(self, x0, sigma0=0.05, boundaries=None):
        super().__init__(x0, sigma0, boundaries)

        # Set optimiser state
        self._running = False
        self._ready_for_tell = False
        self._iterations = 0

        # Best solution found
        self._x_best = self._x0
        self._f_best = np.inf

        # Current point, score, and proposal
        self._current = self._x0
        self._current_f = np.inf
        self._proposed = self._x0
        self._proposed.setflags(write=False)

        # Temperature parameters
        self._temp = 1.0
        self._temp_decay = 0.95
        self._initial_temp = 1.0

    def ask(self):
        """
        Returns a list of next points in the parameter-space
        to evaluate from the optimiser.
        """
        # Update temperature
        self._temp = self._initial_temp * (self._temp_decay**self._iterations)

        # Generate new point with random perturbation
        step = np.random.normal(0, self._sigma0, size=len(self._current))
        self._proposed = self._current + step

        # Apply boundaries
        if self._boundaries is not None:
            self._proposed = np.clip(
                self._proposed, self._boundaries.lower(), self._boundaries.upper()
            )

        # Update state
        self._ready_for_tell = True
        self._running = True

        return [np.array(self._proposed, copy=True)]

    def tell(self, reply):
        """
        Receives a list of function values from the cost function from points
        previously specified by `self.ask()`, and updates the optimiser state
        accordingly.
        """
        if not self._ready_for_tell:
            raise RuntimeError("ask() not called before tell()")
        self._ready_for_tell = False

        # Unpack reply
        fx = reply[0]

        # Accept or reject based on temperature and score.
        # Always accept improved positions, probabilistically
        # accept worse solutions
        if fx < self._current_f:
            accept = True
        else:
            p = np.exp(-(fx - self._current_f) / self._temp)
            accept = np.random.random() < p

        if accept:
            self._current = np.array(self._proposed, copy=True)
            self._current_f = fx

            # Update best if current is best
            if fx < self._f_best:
                self._f_best = fx
                self._x_best = np.array(self._current, copy=True)

        self._iterations += 1

    def name(self):
        """
        Returns the name of this optimiser.
        """
        return "Simulated Annealing"

    def needs_sensitivities(self):
        """
        Returns whether this method needs sensitivities.
        """
        return False

    def n_hyper_parameters(self):
        """
        Returns the number of hyper-parameters for this optimiser.
        """
        return 2

    def running(self):
        """
        Returns whether the optimisation is still running.
        """
        return self._running

    def x_best(self):
        """
        Returns the best position found.
        """
        return self._x_best

    def f_best(self):
        """
        Returns the best score found.
        """
        return self._f_best

    @property
    def temperature(self):
        return self._temp

    @property
    def cooling_rate(self):
        return self._temp_decay

    @cooling_rate.setter
    def cooling_rate(self, alpha):
        """
        Sets the cooling rate for the temperature schedule.
        """
        if not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
            raise ValueError("Cooling rate must be between 0 and 1")
        self._temp_decay = float(alpha)

    @property
    def initial_temperature(self):
        return self._initial_temp

    @initial_temperature.setter
    def initial_temperature(self, temp):
        """
        Sets the initial temperature.
        """
        if not isinstance(temp, (int, float)) or temp <= 0:
            raise ValueError("Initial temperature must be positive")
        self._initial_temp = float(temp)
