import pints


class GradientDescent(pints.GradientDescent):
    """
    Gradient descent optimiser. Inherits from the PINTS gradient descent class.
    """

    def __init__(self, x0, sigma0=0.1, bounds=None):
        if bounds is not None:
            print("Boundaries ignored by GradientDescent")

        boundaries = None  # Bounds ignored in pints.GradDesc
        super().__init__(x0, sigma0, boundaries)


class CMAES(pints.CMAES):
    """
    Class for the PINTS optimisation. Extends the BaseOptimiser class.
    """

    def __init__(self, x0, sigma0=0.1, bounds=None):
        if bounds is not None:
            self.boundaries = pints.RectangularBoundaries(
                bounds["lower"], bounds["upper"]
            )
        else:
            self.boundaries = None

        super().__init__(x0, sigma0, self.boundaries)
