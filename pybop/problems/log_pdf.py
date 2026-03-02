from pybop.costs.base_cost import LogPrior
from pybop.costs.log_likelihoods import LogLikelihood
from pybop.costs.weighted_cost import WeightedCost
from pybop.problems.problem import Problem
from pybop.simulators.base_simulator import BaseSimulator


class LogPDF(Problem):
    """
    A problem that evaluates the log of an (unnormalised) probability density function.
    """

    def __init__(self, simulator: BaseSimulator, cost: LogLikelihood):
        if not isinstance(cost, LogLikelihood):
            raise TypeError("LogPDF requires a LogLikelihood for the cost function.")

        super().__init__(simulator=simulator, cost=cost)


class LogPosterior(LogPDF):
    """
    The log of the proportional posterior, defined as the sum of the log-likelihood and
    the log-prior.
    """

    def __init__(self, simulator: BaseSimulator, cost: LogLikelihood):
        super().__init__(simulator=simulator, cost=cost)

        # Replace the log-likelihood with the log-posterior
        self.log_likelihood = self.cost
        self.log_prior = LogPrior(parameters=self.parameters)
        self._cost = WeightedCost(self.log_likelihood, self.log_prior)  # noqa: SLF001
