from pybop import BaseProblem
from pybop.costs.design_cost import DesignCost
from pybop.parameters.parameter import Parameters


class DesignProblem(BaseProblem):
    """
    Problem class for design optimization problems.

    Extends `BaseProblem` with specifics for applying a model to an experimental design.

    Parameters
    ----------
    simulator : pybop.pybamm.Simulator or pybop.pybamm.EISSimulator
        The model and protocol combined into a simulator object.
    parameters : pybop.Parameter or pybop.Parameters
        An object or list of the parameters for the problem.
    cost : pybop.DesignCost, optional
        Cost function containing the target variable.
    domain : str, optional
        The name of the domain (default: "Time [s]").
    """

    def __init__(
        self,
        simulator,
        parameters: Parameters = None,
        cost: DesignCost = None,
    ):
        super().__init__(simulator, parameters, cost)
