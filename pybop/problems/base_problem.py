import numbers
from typing import Optional

import numpy as np
from SALib.analyze import sobol
from SALib.sample.sobol import sample

from pybop import Parameters


class Problem:
    """
    Defines a callable function `f(x)` that returns the evaluation
    of a cost function. The input `x` is a set of parameters that are
    passed via the `set_params` method. The cost function is evaluated
    using the `run` method.
    """

    def __init__(self, pybop_params: Optional[Parameters] = None):
        if pybop_params is None:
            self._param_names = []
        self._params = pybop_params
        self._param_names = pybop_params.keys()

    def _compute_initial_cost_and_resample(self):
        # Compute the absolute initial cost and resample if required
        x0 = self._params.initial_value()
        self.set_params(x0)
        cost0 = self.run()
        nsamples = 0
        while np.isinf(abs(cost0)) and nsamples < 10:
            x0 = self._params.rvs(apply_transform=True)
            if x0 is None:
                break

            self.set_params(x0)
            cost0 = self.run()
            nsamples += 1
        if nsamples > 0:
            self._params.update(initial_values=x0)

        if np.isinf(np.abs(cost0)):
            raise ValueError("The initial parameter values return an infinite cost.")

    def check_and_store_params(self, p: np.ndarray) -> None:
        """
        Checks if the parameters are valid. p should be a numpy array of one dimensions,
        with length equal to the number of parameters in the model.
        """
        if not isinstance(p, np.ndarray):
            raise TypeError("Parameters must be a numpy array.")
        if p.ndim != 1:
            raise ValueError("Parameters must be a 1D numpy array.")
        if len(p) != len(self._param_names):
            raise ValueError(
                f"Expected {len(self._param_names)} parameters, but got {len(p)}."
            )
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

    def sensitivity_analysis(self, n_samples: int = 256):
        """
        Computes the parameter sensitivities on the cost function using
        SOBOL analyse from the SALib module [1].

        Parameters
        ----------
        n_samples : int, optional
            Number of samples for SOBOL sensitivity analysis,
            performs best as order of 2, i.e. 128, 256, etc.

        References
        ----------
        .. [1] Iwanaga, T., Usher, W., & Herman, J. (2022). Toward SALib 2.0:
               Advancing the accessibility and interpretability of global sensitivity
               analyses. Socio-Environmental Systems Modelling, 4, 18155. doi:10.18174/sesmo.18155

        Returns
        -------
        Sensitivities : dict
        """

        salib_dict = {
            "names": list(self._params.keys()),
            "bounds": self._params.bounds_as_numpy(),
            "num_vars": len(self._params.keys()),
        }

        # Create samples, compute cost
        param_values = sample(salib_dict, n_samples)
        costs = np.empty(param_values.shape[0])
        for i, val in enumerate(param_values):
            self.set_params(val)
            costs[i] = self.run()

        return sobol.analyze(salib_dict, costs)

    def observed_fisher(self, x: np.ndarray) -> np.ndarray:
        """
        Returns the observed Fisher information matrix.
        """
        raise NotImplementedError

    def run(self) -> float:
        """
        Evaluates the underlying simulation and cost function using the
        parameters set in the previous call to `set_params`.
        """
        raise NotImplementedError

    def run_with_sensitivities(
        self,
    ) -> tuple[float, np.ndarray]:
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
