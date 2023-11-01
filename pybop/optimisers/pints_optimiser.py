import pybop
import pints
from pybop.optimisers.base_optimiser import BaseOptimiser
from pints import ErrorMeasure


class PintsOptimiser(BaseOptimiser):
    """
    Wrapper class for the PINTS optimisation class. Extends the BaseOptimiser class.
    """

    def __init__(self, x0, xtol=None, method=None):
        super().__init__()
        self.name = "PINTS Optimiser"

        if method is not None:
            self.method = method
        else:
            self.method = pints.PSO

    def _runoptimise(self, cost_function, x0, bounds):
        """
        Run the PINTS optimisation method.

        Inputs
        ----------
        cost_function: function for optimising
        method: optimisation algorithm
        x0: initialisation array
        bounds: bounds array
        """

        # Wrap bounds
        boundaries = pybop.PintsBoundaries(bounds, x0)

        # Wrap error measure
        error = pybop.PintsError(cost_function, x0)

        # Set up optimisation controller
        controller = pints.OptimisationController(
            error, x0, boundaries=boundaries, method=self.method
        )
        controller.set_max_unchanged_iterations(20)  # default 200

        # Run the optimser
        x, final_cost = controller.run()

        # Get performance statistics
        # output = *pass all output*
        # final_cost
        # num_evals
        output = None
        num_evals = None

        return x, output, final_cost, num_evals


class PintsError(ErrorMeasure):
    """
    An interface class for PyBOP that extends the PINTS ErrorMeasure class.

    From PINTS:
    Abstract base class for objects that calculate some scalar measure of
    goodness-of-fit (for a model and a data set), such that a smaller value
    means a better fit.

    ErrorMeasures are callable objects: If ``e`` is an instance of an
    :class:`ErrorMeasure` class you can calculate the error by calling ``e(p)``
    where ``p`` is a point in parameter space.
    """

    def __init__(self, cost_function, x0):
        self.cost_function = cost_function
        self.x0 = x0

    def __call__(self, x):
        cost = self.cost_function(x)

        return cost

    def evaluateS1(self, x):
        """
        Evaluates this error measure, and returns the result plus the partial
        derivatives of the result with respect to the parameters.

        The returned data has the shape ``(e, e')`` where ``e`` is a scalar
        value and ``e'`` is a sequence of length ``n_parameters``.

        *This is an optional method that is not always implemented.*
        """
        raise NotImplementedError

    def n_parameters(self):
        """
        Returns the dimension of the parameter space this measure is defined
        over.
        """
        return len(self.x0)


class PintsBoundaries(object):
    """
    An interface class for PyBOP that extends the PINTS ErrorMeasure class.

    From PINTS:
    Abstract class representing boundaries on a parameter space.
    """

    def __init__(self, bounds, x0):
        self.bounds = bounds
        self.x0 = x0

    def check(self, parameters):
        """
        Returns ``True`` if and only if the given point in parameter space is
        within the boundaries.

        Parameters
        ----------
        parameters
            A point in parameter space
        """
        result = False
        if (
            parameters[0] >= self.bounds["lower"][0]
            and parameters[1] >= self.bounds["lower"][1]
            and parameters[0] <= self.bounds["upper"][0]
            and parameters[1] <= self.bounds["upper"][1]
        ):
            result = True

        return result

    def n_parameters(self):
        """
        Returns the dimension of the parameter space these boundaries are
        defined on.
        """
        return len(self.x0)

    def sample(self, n=1):
        """
        Returns ``n`` random samples from within the boundaries, for example to
        use as starting points for an optimisation.

        The returned value is a NumPy array with shape ``(n, d)`` where ``n``
        is the requested number of samples, and ``d`` is the dimension of the
        parameter space these boundaries are defined on.

        *Note that implementing :meth:`sample()` is optional, so some boundary
        types may not support it.*

        Parameters
        ----------
        n : int
            The number of points to sample
        """
        raise NotImplementedError
