#
# Extends the Pints' Adam Class with a Weight Decay addition
#

import numpy as np
from pints import Optimiser as PintsOptimiser


class AdamWImpl(PintsOptimiser):
    """
    AdamW optimiser (adaptive moment estimation with weight decay), as described in [1]_.

    This method is an extension of the Adam optimiser that introduces weight decay,
    which helps to regularise the weights and prevent overfitting.

    This class reimplements the Pints' Adam Optimiser, but with the weight decay
    functionality mentioned above. Original creation and credit is attributed to Pints.

    Pseudo-code is given below. Here the value of the j-th parameter at
    iteration i is given as ``p_j[i]`` and the corresponding derivative is
    denoted ``g_j[i]``::

        m_j[i] = beta1 * m_j[i - 1] + (1 - beta1) * g_j[i]
        v_j[i] = beta2 * v_j[i - 1] + (1 - beta2) * g_j[i]**2

        m_j' = m_j[i] / (1 - beta1**(1 + i))
        v_j' = v_j[i] / (1 - beta2**(1 + i))

        p_j[i] = p_j[i - 1] - alpha * (m_j' / (sqrt(v_j') + eps) + lam * p_j[i - 1])

    The initial values of the moments are ``m_j[0] = v_j[0] = 0``, after which
    they decay with rates ``beta1`` and ``beta2``. The default values for these are,
    ``beta1 = 0.9`` and ``beta2 = 0.999``.

    The terms ``m_j'`` and ``v_j'`` are "initialisation bias corrected"
    versions of ``m_j`` and ``v_j`` (see section 2 of the paper).

    The parameter ``alpha`` is a step size, which is set as ``min(sigma0)`` in
    this implementation.

    The parameter ``lam`` is the weight decay rate, which is set to ``0.01``
    by default in this implementation.

    Finally, ``eps`` is a small constant used to avoid division by zero, set to
    ``eps = `np.finfo(float).eps` in this implementation.

    This is an unbounded method: Any ``boundaries`` will be ignored.

    References
    ----------
    .. [1] Decoupled Weight Decay Regularization
           Loshchilov and Hutter, 2019, arxiv (version v3)
           https://doi.org/10.48550/arXiv.1711.05101
    """

    def __init__(self, x0, sigma0=0.015, boundaries=None):
        if boundaries is not None:
            print("NOTE: Boundaries ignored by AdamW")

        self.boundaries = None
        super().__init__(x0, sigma0, self.boundaries)

        # Set optimiser state
        self._running = False
        self._ready_for_tell = False

        # Best solution found
        self._x_best = self._x0
        self._f_best = np.inf

        # Current point, score, and gradient
        self._current = self._x0
        self._current_f = np.inf
        self._current_df = None

        # Proposed next point (read-only, so can be passed to user)
        self._proposed = self._x0
        self._proposed.setflags(write=False)

        # Moment vectors
        self._m = np.zeros(self._x0.shape)
        self._v = np.zeros(self._x0.shape)

        # Exponential decay rates for the moment estimates
        self._b1 = 0.9
        self._b2 = 0.999

        # Step size
        self._alpha = self._sigma0
        # Weight decay rate
        self._lam = 0.01

        # Small number added to avoid divide-by-zero
        self._eps = np.finfo(float).eps

        # Powers of decay rates
        self._b1t = 1
        self._b2t = 1

    def ask(self):
        """
        Returns a list of next points in the parameter-space
        to evaluate from the optimiser.
        """

        # Running, and ready for tell now
        self._ready_for_tell = True
        self._running = True

        # Return proposed points (just the one)
        return [self._proposed]

    def f_best(self):
        """
        Returns the best score found so far.
        """
        return self._f_best

    def f_guessed(self):
        """
        Returns the score of the last guessed point.
        """
        return self._current_f

    def name(self):
        """
        Returns the name of the optimiser.
        """
        return "AdamW"

    def needs_sensitivities(self):
        """
        Returns ``False`` if this optimiser does not require gradient,
        and ``True`` otherwise.
        """
        return True

    def n_hyper_parameters(self):
        """
        The number of hyper-parameters used by this optimiser.
        """
        return 5

    def running(self):
        """
        Returns ``True`` if the optimisation is in progress.
        """
        return self._running

    def tell(self, reply):
        """
        Receives a list of function values from the cost function from points
        previously specified by `self.ask()`, and updates the optimiser state
        accordingly.
        """

        # Check ask-tell pattern
        if not self._ready_for_tell:
            raise RuntimeError("ask() not called before tell()")
        self._ready_for_tell = False

        # Unpack reply
        fx, dfx = reply[0]

        # Update current point
        self._current = self._proposed
        self._current_f = fx
        self._current_df = dfx

        # Update bx^t
        self._b1t *= self._b1
        self._b2t *= self._b2

        # "Update biased first moment estimate"
        self._m = self._b1 * self._m + (1 - self._b1) * dfx

        # "Update biased second raw moment estimate"
        self._v = self._b2 * self._v + (1 - self._b2) * dfx**2

        # "Compute bias-corrected first moment estimate"
        m = self._m / (1 - self._b1t)

        # "Compute bias-corrected second raw moment estimate"
        v = self._v / (1 - self._b2t)

        # Take step with weight decay
        self._proposed = self._current - self._alpha * (
            m / (np.sqrt(v) + self._eps) + self._lam * self._current
        )

        # Update x_best and f_best
        if self._f_best > fx:
            self._f_best = fx
            self._x_best = self._current

    def x_best(self):
        """
        Returns the best parameter values found so far.
        """
        return self._x_best

    def x_guessed(self):
        """
        Returns the last guessed parameter values.
        """
        return self._current

    @property
    def lam(self):
        return self._lam

    @lam.setter
    def lam(self, lam: float = 0.01) -> None:
        """
        Sets the lam decay constant. This is the weight decay rate
        that helps in finding the optimal solution.
        """
        if not isinstance(lam, (int, float)) or not 0 < lam <= 1:
            raise ValueError("lam must be a numeric value between 0 and 1.")

        self._lam = float(lam)

    @property
    def b1(self):
        return self._b1

    @b1.setter
    def b1(self, b1: float) -> None:
        """
        Sets the b1 momentum decay constant.
        """
        if not isinstance(b1, (int, float)) or not 0 < b1 <= 1:
            raise ValueError("b1 must be a numeric value between 0 and 1.")

        self._b1 = float(b1)

    @property
    def b2(self):
        return self._b2

    @b2.setter
    def b2(self, b2: float) -> None:
        """
        Sets the b2 momentum decay constant.
        """
        if not isinstance(b2, (int, float)) or not 0 < b2 <= 1:
            raise ValueError("b2 must be a numeric value between 0 and 1.")

        self._b2 = float(b2)
