from pybop._utils import FailedSolution
from pybop.costs.base_cost import BaseCost
from pybop.parameters.parameter import Inputs
from pybop.simulators.base_simulator import Solution


class DesignCost(BaseCost):
    """
    Base design cost.

    Note that design costs are maximised by default. Change to minimising by setting
    the attribute `minimising=True`.

    Parameters
    ----------
    target : str
        The name of the target variable.
    """

    def __init__(self, target: str):
        super().__init__()
        self.minimising = False
        target = [target] if isinstance(target, str) else target
        self.target = target or ["Voltage [V]"]
        self.domain = "Time [s]"

    def evaluate(
        self,
        sol: Solution | FailedSolution,
        inputs: Inputs | None = None,
        calculate_sensitivities: bool = False,
    ) -> float:
        """
        Returns the value of the cost variable.

        Parameters
        ----------
        sol : pybop.Solution | pybamm.Solution
            The simulation result.
        inputs : Inputs, optional
            Input parameters (default: None).
        calculate_sensitivities : bool
            Whether to also return the sensitivities (default: False).

        Returns
        -------
        float
            The value of the output variable.
        """
        # Return failure cost if the solution failed
        if isinstance(sol, FailedSolution):
            return self.failure(calculate_sensitivities)

        return sol[self.target[0]].data[-1]
