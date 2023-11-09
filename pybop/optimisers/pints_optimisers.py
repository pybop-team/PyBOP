import pints


class GradientDescent(pints.GradientDescent):
    """
    Gradient descent optimiser. Inherits from the PINTS gradient descent class.
    """

    def __init__(self, x0, sigma0=0.1, bounds=None):
        boundaries = PintsBoundaries(bounds, x0)
        super().__init__(x0, sigma0, boundaries)


class CMAES(pints.CMAES):
    """
    Class for the PINTS optimisation. Extends the BaseOptimiser class.
    """

    def __init__(self, x0, sigma0=0.1, bounds=None):
        boundaries = PintsBoundaries(bounds, x0)
        super().__init__(x0, sigma0, boundaries)


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

    # def sample(self, n=1):
    #     """
    #     Returns ``n`` random samples from within the boundaries, for example to
    #     use as starting points for an optimisation.

    #     The returned value is a NumPy array with shape ``(n, d)`` where ``n``
    #     is the requested number of samples, and ``d`` is the dimension of the
    #     parameter space these boundaries are defined on.

    #     *Note that implementing :meth:`sample()` is optional, so some boundary
    #     types may not support it.*

    #     Parameters
    #     ----------
    #     n : int
    #         The number of points to sample
    #     """
    #     raise NotImplementedError
