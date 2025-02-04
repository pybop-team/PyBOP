#
# Initially based of Pints' IRProp- class.
#

import numpy as np
from pints import Optimiser as PintsOptimiser
from pints import RectangularBoundaries


class IRPropPlusImpl(PintsOptimiser):
    """
    The iRprop+ algorithm adjusts step sizes based on the sign of the gradient
    in each dimension, increasing step sizes when the sign remains consistent
    and decreasing when the sign changes. This implementation includes
    weight (parameter) backtracking that reverts the previous step in the
    event of a gradient sign changing.

    References:
    - [1] Igel and Hüsken (2003): Empirical Evaluation of Improved Rprop Algorithms.
    - [2] Riedmiller and Braun (1993): A Direct Adaptive Method for Faster Backpropagation.
    - [3] Igel and Husk (2003): Improving the Rprop Learning Algorithm.

    Parameters
    ----------
    x0 : array-like
        Initial starting point for the optimisation.
    sigma0 : float or array-like, optional
        Initial step size(s). If a scalar is provided, it is applied to all dimensions.
        Default is 0.05.
    boundaries : pints.Boundaries, optional
        Boundary constraints for the optimisation. If None, no boundaries are applied.

    Attributes
    ----------
    eta_min : float
        Factor by which the step size is reduced when the gradient sign changes.
        Default is 0.5.
    eta_max : float
        Factor by which the step size is increased when the gradient sign remains consistent.
        Default is 1.2.
    step_min : float
        Minimum allowable step size. Default is 1e-3 * min(sigma0).
    step_max : float or None
        Maximum allowable step size. Default is None (unlimited).
    """

    def __init__(self, x0, sigma0=0.05, boundaries=None):
        super().__init__(x0, sigma0, boundaries)

        # Set hypers
        self.eta_min = 0.5
        self.eta_max = 1.2
        self.step_min = 1e-4 * np.min(self._sigma0)
        self.step_max = None

        # Store the previous update for backtracking
        self._update_previous = np.zeros_like(x0, dtype=float)

        # Internal states
        self._x_current = np.array(x0, dtype=float)
        self._gradient_previous = None
        self._step_sizes = np.copy(self._sigma0)
        self._f_current = np.inf
        self._x_best = np.array(x0, dtype=float)
        self._f_best = np.inf
        self._running = False
        self._ready_for_tell = False

        # Boundaries
        self._boundaries = boundaries
        if isinstance(boundaries, RectangularBoundaries):
            self._lower = boundaries.lower()
            self._upper = boundaries.upper()
        else:
            self._lower = self._upper = None

        # Set proposals
        self._proposed = np.copy(self._x_current)
        self._proposed.setflags(write=False)

    def ask(self):
        """
        Proposes the next point for evaluation.

        Returns
        -------
        list
            A list containing the proposed point.
        """
        if not self._running:
            if self.step_min is not None and self.step_max is not None:
                if self.step_min >= self.step_max:
                    raise ValueError(
                        f"Minimum step size ({self.step_min}) must be less than "
                        f"maximum step size ({self.step_max})."
                    )
            self._running = True

        self._ready_for_tell = True
        return [self._proposed]

    def tell(self, reply):
        """
        Updates the optimiser with the function value and gradient at the proposed point.

        Parameters
        ----------
        reply : list
            A list containing a tuple of (function value, gradient) at the proposed point.

        Raises
        ------
        RuntimeError
            If `ask()` was not called before `tell()`.
        """
        if not self._ready_for_tell:
            raise RuntimeError("ask() must be called before tell().")

        self._ready_for_tell = False
        f_new, gradient_new = reply[0]

        # Setup for first iteration
        if self._gradient_previous is None:
            self._gradient_previous = gradient_new
            self._f_current = f_new
            return

        # Compute element-wise gradient product
        grad_product = gradient_new * self._gradient_previous

        # Update step sizes, and bound them
        self._step_sizes[grad_product > 0] *= self.eta_max
        self._step_sizes[grad_product < 0] *= self.eta_min
        self._step_sizes = np.clip(self._step_sizes, self.step_min, self.step_max)

        # Handle weight backtracking,
        # Reverting last update where gradient sign changed
        gradient_new[grad_product < 0] = 0
        self._x_current[grad_product < 0] -= self._update_previous[grad_product < 0]

        # Update the current position
        self._x_current = np.copy(self._proposed)
        self._f_current = f_new
        self._gradient_previous = gradient_new

        # Compute the new update and store for back-tracking
        update = -self._step_sizes * np.sign(gradient_new)
        self._update_previous = update

        # Step in the direction of the negative gradient
        proposed = self._x_current + update

        # Boundaries
        if self._lower is not None:
            # Rectangular boundaries
            while np.any(proposed < self._lower) or np.any(proposed >= self._upper):
                mask = np.logical_or(proposed < self._lower, proposed >= self._upper)
                self._step_sizes[mask] *= self.eta_min
                proposed = self._x_current - self._step_sizes * np.sign(gradient_new)

        # Update proposed attribute
        self._proposed = proposed
        self._proposed.setflags(write=False)

        # Update best solution
        if f_new < self._f_best:
            self._f_best = f_new
            self._x_best = self._x_current

    def running(self):
        """Returns the state of the optimiser"""
        return self._running

    def f_best(self):
        """Returns the best function value found so far."""
        return self._f_best

    def x_best(self):
        """Returns the best position found so far."""
        return self._x_best

    def name(self):
        """Returns the name of the optimiser."""
        return "iRprop+"

    def needs_sensitivities(self):
        """Indicates that this optimiser requires gradient information."""
        return True
