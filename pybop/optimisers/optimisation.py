from pybop import XNES, BasePintsOptimiser, BaseSciPyOptimiser


class Optimisation:
    """
    A high-level class for optimisation using PyBOP or PINTS optimisers.

    This class provides an alternative API to the `PyBOP.Optimiser()` API,
    specifically allowing for single user-friendly interface for the
    optimisation process.The class can be used with either PyBOP or PINTS
    optimisers.

    Parameters
    ----------
    cost : pybop.BaseCost or pints.ErrorMeasure
        An objective function to be optimized, which can be either a pybop.Cost
    optimiser : pybop.Optimiser or subclass of pybop.BaseOptimiser, optional
        An optimiser from either the PINTS or PyBOP framework to perform the optimisation (default: None).
    sigma0 : float or sequence, optional
        Initial step size or standard deviation for the optimiser (default: None).
    verbose : bool, optional
        If True, the optimisation progress is printed (default: False).
    physical_viability : bool, optional
        If True, the feasibility of the optimised parameters is checked (default: True).
    allow_infeasible_solutions : bool, optional
        If True, infeasible parameter values will be allowed in the optimisation (default: True).

    Attributes
    ----------
    All attributes from the pybop.optimiser() class

    """

    def __init__(self, cost, optimiser=None, **optimiser_kwargs):
        self.__dict__["optim"] = (
            None  # Pre-define optimiser to avoid recursion during initialisation
        )
        if optimiser is None:
            self.optim = XNES(cost, **optimiser_kwargs)
        elif issubclass(optimiser, BasePintsOptimiser):
            self.optim = optimiser(cost, **optimiser_kwargs)
        elif issubclass(optimiser, BaseSciPyOptimiser):
            self.optim = optimiser(cost, **optimiser_kwargs)
        else:
            raise ValueError("Unknown optimiser type")

    def run(self):
        return self.optim.run()

    def __getattr__(self, attr):
        if "optim" in self.__dict__ and hasattr(self.optim, attr):
            return getattr(self.optim, attr)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{attr}'"
        )

    def __setattr__(self, name: str, value) -> None:
        if (
            name in self.__dict__
            or "optim" not in self.__dict__
            or not hasattr(self.optim, name)
        ):
            object.__setattr__(self, name, value)
        else:
            setattr(self.optim, name, value)
