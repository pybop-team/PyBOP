from pybop import BaseProblem
from pybop.costs.error_measures import ErrorMeasure
from pybop.costs.likelihoods import LogLikelihood
from pybop.parameters.parameter import Parameters


class FittingProblem(BaseProblem):
    """
    Problem class for fitting (parameter estimation) problems.

    Extends `BaseProblem` with specifics for fitting a model to a dataset.

    Parameters
    ----------
    simulator : pybop.pybamm.Simulator or pybop.pybamm.EISSimulator
        The model, protocol and dataset combined into a simulator object.
    parameters : pybop.Parameter or pybop.Parameters
        An object or list of the parameters for the problem.
    cost : pybop.ErrorMeasure | pybop.LogLikelihood, optional
        Cost function containing the target data.

    Additional
    """

    def __init__(
        self,
        simulator,
        parameters: Parameters = None,
        cost: ErrorMeasure | LogLikelihood | None = None,
    ):
        super().__init__(simulator, parameters, cost)
