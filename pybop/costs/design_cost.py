import numpy as np

from pybop.costs.base_cost import BaseCost
from pybop.costs.evaluation import Evaluation
from pybop.parameters.parameter import Inputs
from pybop.simulators.base_simulator import Solution
from pybop.simulators.failed_solution import FailedSolution


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
        self.set_target(target)

    def __call__(
        self,
        solution: Solution | FailedSolution,
        inputs: Inputs | None = None,
        calculate_sensitivities: bool = False,
    ) -> float:
        """
        Compute the value of the cost variable.

        Parameters
        ----------
        solution : pybop.Solution | pybamm.Solution
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
        if isinstance(solution, FailedSolution):
            return self.failure(self.parameters.names, calculate_sensitivities)

        return solution[self.target[0]].data[-1]

    def evaluate_batch(
        self,
        solution: list[Solution],
        inputs: list[Inputs],
        calculate_sensitivities: bool = False,
    ) -> Evaluation:
        """
        Returns the value of the cost variable.

        Parameters
        ----------
        solution : list[Solution]
            A list of simulation results.
        inputs : list[Inputs]
            The corresponding list of input parameters.
        calculate_sensitivities : bool
            Whether to also return the sensitivities (default: False).
        """
        e = np.empty(len(solution))

        for i, (sol, x) in enumerate(zip(solution, inputs, strict=False)):
            e[i] = self.__call__(solution=sol, inputs=x)

        return Evaluation(values=e)
