import numpy as np
from pints import Optimiser as PintsOptimiser


class GradientDescentImpl(PintsOptimiser):
    """
    Gradient descent method with a fixed, per-dimension learning rate.

    Gradient descent updates the current position in the direction of the
    steepest descent, as determined by the negative of the gradient of the
    function.

    The update rule for each iteration is given by:

    .. math::

        x_{t+1} = x_t - \\eta * \\nabla f(x_t)

    where:
        - :math:`x_t` are the current parameter values at iteration t,
        - :math:`\\nabla f(x_t)` is the gradient of the function at :math:`x_t`,
        - :math:`\\eta` is the learning rate, which controls the step size.

    This class reimplements the Pints' Gradient Descent, but with multidimensional,
    fixed learning rates. Original creation and credit is attributed to Pints.

    Parameters
    ----------
    x0 : array-like
        Initial starting point for the optimisation. This should be a 1D array
        representing the starting parameter values for the function being
        optimised.
    sigma0 : float or array-like, optional
        Initial learning rate or rates for each dimension. If a scalar is
        provided, the same learning rate is applied across all dimensions.
        If an array is provided, each dimension will have its own learning
        rate. Defaults to 0.02.
    boundaries : pybop.Boundaries, optional
        Boundaries for the parameters. This optimiser ignores boundaries and
        operates as an unbounded method. Defaults to None.

    Attributes
    ----------
    _x_best : array-like
        The best parameter values (solution) found so far.
    _f_best : float
        The best function value (objective value) found so far.
    _current : array-like
        The current parameter values at the latest iteration.
    _eta : array-like
        The current learning rate(s). Can be a scalar or per-dimension array.
    _running : bool
        Indicates whether the optimisation process is running.
    _ready_for_tell : bool
        Indicates whether the optimiser is ready to receive feedback from
        the objective function.
    """

    def __init__(self, x0, sigma0=0.02, boundaries=None):
        super().__init__(x0, sigma0, boundaries)

        # Initialise state
        self._x_best = self._current = self._x0
        self._f_best = np.inf
        self._eta = np.asarray(sigma0, dtype=float)

        # State tracking
        self._running = False
        self._ready_for_tell = False

    def ask(self):
        """Proposes the next point for evaluation."""
        self._ready_for_tell = True
        self._running = True
        return [self._current]

    def tell(self, reply):
        """Updates optimiser with function evaluation results."""
        if not self._ready_for_tell:
            raise RuntimeError("ask() must be called before tell().")
        self._ready_for_tell = False

        fx, dfx = reply[0]

        # Update state
        self._current_f, self._current_df = fx, dfx
        self._current = self._current - self._eta * dfx

        # Track best solution
        if fx < self._f_best:
            self._f_best, self._x_best = fx, self._current

    def f_best(self):
        """Returns the best objective value found."""
        return self._f_best

    def x_best(self):
        """Returns the best solution found."""
        return self._x_best

    def learning_rate(self):
        """Returns the learning rate(s)."""
        return self._eta

    def set_learning_rate(self, eta):
        """
        Sets the learning rate. Supports per-dimension rates.

        Parameters
        ----------
        eta : float or array-like
            New learning rate(s).
        """
        eta = np.asarray(eta, dtype=float)
        if np.any(eta <= 0):
            raise ValueError("Learning rate(s) must be positive.")
        self._eta = eta

    def needs_sensitivities(self):
        """Indicates this optimiser requires gradient information."""
        return True

    def running(self):
        """Returns whether the optimiser is running."""
        return self._running

    def name(self):
        """Returns the name of the optimiser."""
        return "Gradient descent"

    def n_hyper_parameters(self):
        """Returns the number of hyper-parameters (learning rate)."""
        return self._eta.size if self._eta.ndim > 0 else 1

    def set_hyper_parameters(self, x):
        """Sets hyper-parameters (learning rate)."""
        self.set_learning_rate(x)
