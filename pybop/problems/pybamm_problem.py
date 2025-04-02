import pybop
import numpy as np
import pybamm


class PybammProblem(pybamm.Problem):
    """
    Defines a problem that uses a PyBaMM model as the simulation + cost function to evaluate
    """

    def __init__(
        self,
        pybamm_pipeline: pybop.PyBaMMPipeline,
        param_names: list[str] = None,
        cost_names: list[str] = None,
    ):
        super().__init__(param_names=param_names)
        self._pipeline = pybamm_pipeline
        self._cost_names = cost_names

    def set_params(self, p: np.ndarray) -> None:
        """
        Sets the parameters for the simulation and cost function.
        """
        self.check_and_store_params(p)

        # rebuild the pipeline (if needed)
        self._pipeline.rebuild(self._params)

    def run(self) -> float:
        """
        Evaluates the underlying simulation and cost function using the
        parameters set in the previous call to `set_params`.
        """
        self.check_set_params_called()

        # run simulation
        sol = self._pipeline.solve()

        # extract and sum cost function values. These are assumed to all be scalar values
        # (not to self: test this is true in tests....)
        cost = sum([sol[n].values[0] for n in self._cost_names])
        return cost

    def run_with_sensitivities(
        self,
    ) -> tuple[float, np.ndarray]:
        """
        Evaluates the underlying simulation and cost function using the
        parameters set in the previous call to `set_params`.
        """
        self.check_set_params_called()

        # run simulation
        sol = self._pipeline.solve(calculate_sensitivities=True)

        # extract cost function values. These are assumed to all be scalar values
        # (not to self: test this is true in tests....)
        cost = sum([sol[n].values[0] for n in self._cost_names])

        # sensitivities will all be 1D arrays of length n_params, sum over the different
        # cost functions to get the total sensitivity
        cost_sens = np.array(
            [
                sum([sol[cost_n].sensitivities[param_n] for cost_n in self._cost_names])
                for param_n in self._params
            ]
        )
        return cost, cost_sens
