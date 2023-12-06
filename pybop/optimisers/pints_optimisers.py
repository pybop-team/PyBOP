import pints


class GradientDescent(pints.GradientDescent):
    """
    Gradient descent optimiser. Inherits from the PINTS gradient descent class.
    https://github.com/pints-team/pints/blob/main/pints/_optimisers/_gradient_descent.py
    """

    def __init__(self, x0, sigma0=0.1, bounds=None):
        if bounds is not None:
            print("NOTE: Boundaries ignored by Gradient Descent")

        self.boundaries = None  # Bounds ignored in pints.GradDesc
        super().__init__(x0, sigma0, self.boundaries)


class Adam(pints.Adam):
    """
    Adam optimiser. Inherits from the PINTS Adam class.
    https://github.com/pints-team/pints/blob/main/pints/_optimisers/_adam.py
    """

    def __init__(self, x0, sigma0=0.1, bounds=None):
        if bounds is not None:
            print("NOTE: Boundaries ignored by Adam")

        self.boundaries = None  # Bounds ignored in pints.Adam
        super().__init__(x0, sigma0, self.boundaries)


class IRPropMin(pints.IRPropMin):
    """
    IRProp- optimiser. Inherits from the PINTS IRPropMinus class.
    https://github.com/pints-team/pints/blob/main/pints/_optimisers/_irpropmin.py
    """

    def __init__(self, x0, sigma0=0.1, bounds=None):
        if bounds is not None:
            self.boundaries = pints.RectangularBoundaries(
                bounds["lower"], bounds["upper"]
            )
        else:
            self.boundaries = None
        super().__init__(x0, sigma0, self.boundaries)


class PSO(pints.PSO):
    """
    Particle swarm optimiser. Inherits from the PINTS PSO class.
    https://github.com/pints-team/pints/blob/main/pints/_optimisers/_pso.py
    """

    def __init__(self, x0, sigma0=0.1, bounds=None):
        if bounds is not None:
            self.boundaries = pints.RectangularBoundaries(
                bounds["lower"], bounds["upper"]
            )
        else:
            self.boundaries = None
        super().__init__(x0, sigma0, self.boundaries)


class SNES(pints.SNES):
    """
    Stochastic natural evolution strategy optimiser. Inherits from the PINTS SNES class.
    https://github.com/pints-team/pints/blob/main/pints/_optimisers/_snes.py
    """

    def __init__(self, x0, sigma0=0.1, bounds=None):
        if bounds is not None:
            self.boundaries = pints.RectangularBoundaries(
                bounds["lower"], bounds["upper"]
            )
        else:
            self.boundaries = None
        super().__init__(x0, sigma0, self.boundaries)


class XNES(pints.XNES):
    """
    Exponential natural evolution strategy optimiser. Inherits from the PINTS XNES class.
    https://github.com/pints-team/pints/blob/main/pints/_optimisers/_xnes.py
    """

    def __init__(self, x0, sigma0=0.1, bounds=None):
        if bounds is not None:
            self.boundaries = pints.RectangularBoundaries(
                bounds["lower"], bounds["upper"]
            )
        else:
            self.boundaries = None
        super().__init__(x0, sigma0, self.boundaries)


class CMAES(pints.CMAES):
    """
    Class for the PINTS optimisation. Extends the BaseOptimiser class.
    https://github.com/pints-team/pints/blob/main/pints/_optimisers/_cmaes.py
    """

    def __init__(self, x0, sigma0=0.1, bounds=None):
        if bounds is not None:
            self.boundaries = pints.RectangularBoundaries(
                bounds["lower"], bounds["upper"]
            )
        else:
            self.boundaries = None

        super().__init__(x0, sigma0, self.boundaries)
