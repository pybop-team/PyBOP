import numpy as np

from pybop import Parameters
from pybop.analysis.sensitivity_analysis import sensitivity_analysis


class Problem:
    """
    Defines a `run` method that takes candidate parameter sets returns the corresponding
    values of the cost and sensitivities if needed.
    """

    def __init__(self, pybop_params: Parameters | None = None):
        if pybop_params is None:
            self._param_names = []
        self._params = pybop_params
        self._param_names = pybop_params.keys()
        self._has_sensitivities = False

    def get_finite_initial_cost(self):
        """
        Compute the absolute initial cost, resampling the initial parameters if needed.
        """
        x0 = self._params.get_initial_values()
        cost0 = np.abs(self.run(x0))
        nsamples = 0
        while np.isinf(cost0) and nsamples < 10:
            x0 = self._params.sample_from_priors()
            if x0 is None:
                break

            cost0 = np.abs(self.run(x0))
            nsamples += 1
        if nsamples > 0:
            self._params.update(initial_values=x0)

        if np.isinf(cost0):
            raise ValueError("The initial parameter values return an infinite cost.")
        return cost0

    @property
    def params(self) -> Parameters:
        """
        Returns the parameters set for the simulation and cost function.
        """
        return self._params

    @property
    def param_names(self) -> list[str]:
        """
        Returns the names of the parameters set for the simulation and cost function.
        """
        return self._param_names

    @property
    def has_sensitivities(self) -> bool:
        return self._has_sensitivities

    def sensitivity_analysis(self, n_samples: int = 256) -> dict:
        """
        Computes the parameter sensitivities on the combined cost function using
        SOBOL analysis. See pybop.analysis.sensitivity_analysis for more details.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples for SOBOL sensitivity analysis, performs best as a
            power of 2, i.e. 128, 256, etc.
        """
        return sensitivity_analysis(problem=self, n_samples=n_samples)

    def run(self, p) -> np.ndarray:
        """
        Evaluates the underlying simulation and cost function.
        """
        raise NotImplementedError

    def run_with_sensitivities(self, p) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluates the underlying simulation and cost function, returns
        the cost and sensitivities.
        """
        raise NotImplementedError
