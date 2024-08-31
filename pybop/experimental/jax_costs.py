from typing import Union

import jax
import jax.numpy as jnp
import numpy as np

from pybop import BaseCost, BaseLikelihood, BaseProblem, Inputs


class BaseJaxCost(BaseCost):
    """
    Jax-based Sum of Squared Error cost function.
    """

    def __init__(self, problem: BaseProblem):
        super().__init__(problem)

    def __call__(
        self,
        inputs: Inputs,
        calculate_grad: bool = False,
    ) -> Union[np.array, tuple[float, np.ndarray]]:
        """
        Computes the cost function for the given predictions.

        Parameters
        ----------
        y : dict
            The dictionary of predictions with keys designating the signals for fitting.
        dy : np.ndarray, optional
            The corresponding gradient with respect to the parameters for each signal.
        calculate_grad : bool, optional
            A bool condition designating whether to calculate the gradient.

        Returns
        -------
        float
            The Sum of Squared Error.
        """
        inputs = self.parameters.verify(inputs)
        self._update_solver_sensitivities(calculate_grad)

        if calculate_grad:
            y, dy = jax.value_and_grad(self.evaluate)(inputs)
            return y, np.stack(list(dy.values()))
        else:
            return self.evaluate(inputs)

    def _update_solver_sensitivities(self, calculate_grad: bool) -> None:
        """
        Updates the solver's sensitivity calculation based on the gradient requirement.

        Args:
            calculate_grad (bool): Whether gradient calculation is required.
        """
        model = self.problem.model
        if calculate_grad != model.calculate_sensitivities:
            model.jaxify_solver(
                t_eval=self.problem.domain_data, calculate_sensitivities=calculate_grad
            )

    @staticmethod
    def check_sigma0(sigma0):
        if sigma0 is None:
            return 0.005
        if not isinstance(sigma0, (int, float)) or sigma0 <= 0:
            raise ValueError("sigma0 must be a positive number")
        return float(sigma0)


class JaxSumSquaredError(BaseJaxCost):
    """
    Jax-based Sum of Squared Error cost function.
    """

    def __init__(self, problem: BaseProblem):
        super().__init__(problem)

    def evaluate(self, inputs):
        # Calculate residuals and error
        y = self.problem.jax_evaluate(inputs)
        r = jnp.asarray([y - self._target[signal] for signal in self.signal])
        return jnp.sum(jnp.sum(r**2, axis=0), axis=0)


class JaxLogNormalLikelihood(BaseJaxCost, BaseLikelihood):
    """
    A Log-Normal Likelihood function. This function represents the
    underlining observed data sampled from a Log-Normal distribution.
    """

    def __init__(self, problem: BaseProblem, sigma0=None):
        super().__init__(problem)
        self.sigma = self.check_sigma0(sigma0)
        self.sigma2 = jnp.square(self.sigma)
        self._offset = 0.5 * self.n_data * jnp.log(2 * jnp.pi)
        self._target_as_array = jnp.asarray(
            [self._target[signal] for signal in self.signal]
        )
        self._log_target_sum = jnp.sum(jnp.log(self._target_as_array))
        self._precompute()

    def _precompute(self):
        self._constant_term = (
            -self._offset - self.n_data * jnp.log(self.sigma) - self._log_target_sum
        )

    def evaluate(self, inputs):
        """
        Evaluates the log-normal likelihood.
        """
        y = self.problem.jax_evaluate(inputs)
        e = jnp.log(self._target_as_array) - jnp.log(y)
        return self._constant_term - jnp.sum(jnp.square(e)) / (2 * self.sigma2)
