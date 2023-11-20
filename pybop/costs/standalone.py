import pybop
import numpy as np


class StandaloneCost(pybop.BaseCost):
    def __init__(self, problem=None):
        super().__init__(problem)

        self.x0 = np.array([4.2])
        self.n_parameters = len(self.x0)

        self.bounds = dict(
            lower=[-1],
            upper=[10],
        )

    def __call__(self, x, grad=None):
        return x[0] ** 2 + 42
