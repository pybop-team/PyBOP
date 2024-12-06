import numpy as np
from pybop.costs.base_cost import BaseCost
from pybop.problems.base_problem import BaseProblem
import pybamm


class PybammLogNormalLikelihood(BaseCost):
    """
    A Log-Normal Likelihood function. This function represents the
    underlining observed data sampled from a Log-Normal distribution.
    Parameters
    -----------
    problem: BaseProblem
        The problem to fit of type `pybop.BaseProblem`
    """

    def __init__(self, problem: BaseProblem):
        super().__init__(problem)

        # Define the log-normal likelihood
        sigma = pybamm.InputParameter("sigma")
        sigma2 = sigma**2

        data_times = problem.dataset["Time [s]"]
        log_data_values = np.asarray([np.log(self._target[s]) for s in self.signal]).T
        log_data = pybamm.DiscreteTimeData(
            data_times, log_data_values, "log-normal-data"
        )
        if not problem.model.built_model:
            raise ValueError("Model must be built before using this cost function")

        pybamm_model = problem.model._built_model
        if len(self.signal) > 1:
            log_signals = pybamm.log(
                pybamm.numpy_concatenation(
                    *[pybamm_model.variables[s] for s in self.signal]
                )
            )
        else:
            log_signals = pybamm.log(pybamm_model.variables[self.signal[0]])
        likelihood = pybamm.DiscreteTimeSum(
            (log_signals - log_data) ** 2 / (2 * sigma2)
        )

        # need this due to https://github.com/pybamm-team/PyBaMM/issues/4637
        likelihood.child.mesh = None

        # add to the model variables
        self._my_variable_name = f"log-normal-likelihood-{'-'.join(self.signal)}"
        pybamm_model.variables.update({self._my_variable_name: likelihood})
        problem.additional_variables.append(self._my_variable_name)

    def compute(self, y: dict, dy: np.ndarray, calculate_grad: bool = False):
        print(y)
        value = y[self._my_variable_name]
        if calculate_grad:
            grad = y[self._my_variable_name].sensitivities.data
            return value, grad
        return value
