import jax

from pybop import BaseProblem, Inputs


class JaxSumSquaredError:
    """
    Jax-based Sum of Squared Error cost function.
    """

    def __init__(self, problem: BaseProblem):
        self._target = problem.target
        self._problem = problem

    @jax.jit
    def __call__(self, inputs: Inputs):
        """Evaluate the jax-based cost function"""
        # TODO: Jaxified FittingProblem -> Can BaseProblem be @jitted?
