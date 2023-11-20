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
        if bounds is not None:
            boundaries = pints.RectangularBoundaries(bounds["lower"], bounds["upper"])
        else:
            boundaries = None  # PintsBoundaries(bounds, x0)

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
        """

        lower_bounds = self.bounds["lower"]
        upper_bounds = self.bounds["upper"]

        if len(parameters) != len(lower_bounds):
            raise ValueError("Parameters length mismatch")

        within_bounds = all(
            low <= param <= high
            for low, high, param in zip(lower_bounds, upper_bounds, parameters)
        )

        return within_bounds

    def n_parameters(self):
        """
        Returns the dimension of the parameter space these boundaries are
        defined on.
        """
        return len(self.x0)
