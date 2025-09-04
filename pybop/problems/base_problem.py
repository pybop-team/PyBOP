import numpy as np

from pybop import Parameters
from pybop.analysis.sensitivity_analysis import sensitivity_analysis


class Problem:
    """
    Defines a callable function `f(x)` that returns the evaluation
    of a cost function. The input `x` is a set of parameters that are
    passed via the `set_params` method. The cost function is evaluated
    using the `run` method.
    """

    def __init__(self, pybop_params: Parameters | None = None):
        if pybop_params is None:
            self._param_names = []
        self._params = pybop_params
        self._param_names = pybop_params.keys()

    def get_finite_initial_cost(self):
        """
        Compute the absolute initial cost, resampling the initial parameters if needed.
        """
        x0 = self._params.get_initial_values()
        self.set_params(x0)
        cost0 = np.abs(self.run())
        nsamples = 0
        while np.isinf(cost0) and nsamples < 10:
            x0 = self._params.sample_from_priors()
            if x0 is None:
                break

            self.set_params(x0)
            cost0 = np.abs(self.run())
            nsamples += 1
        if nsamples > 0:
            self._params.update(initial_values=x0)

        if np.isinf(cost0):
            raise ValueError("The initial parameter values return an infinite cost.")
        return cost0

    def check_and_store_params(self, p: np.ndarray) -> None:
        """
        Checks if the parameters are valid. p should be a numpy array of one dimensions,
        with length equal to the number of parameters in the model.
        """
        if not isinstance(p, np.ndarray | list):
            raise TypeError("Parameters must be a numpy array or list")
        if isinstance(p, list):
            try:
                p = np.asarray(p)
            except TypeError as e:
                raise TypeError(
                    "Parameters cannot be converted to a numpy array."
                ) from e
        self._params.update(values=p)

    def check_set_params_called(self) -> None:
        """
        Checks if the parameters have been set.
        """
        if self._params is None:
            raise ValueError(
                "Parameters have not been set. Call `set_params` before running the simulation."
            )

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
        return NotImplementedError

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

    def run(self) -> np.ndarray:
        """
        Evaluates the underlying simulation and cost function using the
        parameters set in the previous call to `set_params`.
        """
        raise NotImplementedError

    def run_with_sensitivities(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluates the underlying simulation and cost function using the
        parameters set in the previous call to `set_params`, returns
        the cost and sensitivities.
        """
        raise NotImplementedError

    def set_params(self, p: np.ndarray) -> None:
        """
        Sets the parameters for the simulation and cost function.
        The arg `p` is a numpy array of parameters in the model space.
        Hint: use `to_search` and `from_search` to convert between model and
        search space.
        """
        raise NotImplementedError
