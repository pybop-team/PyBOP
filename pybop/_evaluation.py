import jax.numpy as jnp
import numpy as np
from pints import Evaluator as PintsEvaluator


class SequentialJaxEvaluator(PintsEvaluator):
    """
    Sequential evaluates a function (or callable object)
    for either a single or multiple positions. This class is based
    off the PintsSequentialEvaluator class, with additions for
    PyBOP's JAX cost classes.

    Parameters
    ----------
    function : callable
        The function to evaluate. This function should accept an input and
        optionally additional arguments, returning either a single value or a tuple.
    args : sequence, optional
        A sequence containing extra arguments to be passed to the function.
        If specified, the function will be called as `function(x, *args)`.
    """

    def _evaluate(self, positions):
        scores = [self._function(x, *self._args) for x in positions]

        # If gradient provided, convert jnp to np and return
        if isinstance(scores[0], tuple):
            return [(score[0].item(), score[1]) for score in scores]

        return np.asarray(scores)


class SciPyEvaluator(PintsEvaluator):
    """
    Evaluates a function (or callable object) for the SciPy optimisers
    for either a single or multiple positions.

    Parameters
    ----------
    function : callable
        The function to evaluate. This function should accept an input and
        optionally additional arguments, returning either a single value or a tuple.
    args : sequence, optional
        A sequence containing extra arguments to be passed to the function.
        If specified, the function will be called as `function(x, *args)`.
    """

    def _evaluate(self, positions):
        scores = [self._function(x, *self._args) for x in [positions]]

        if not isinstance(scores[0], tuple):
            return np.asarray(scores)[0]

        # If gradient provided, convert jnp to np and return
        if isinstance(scores[0][0], jnp.ndarray):
            return [(score[0].item(), score[1]) for score in scores][0]
        return [(score[0], score[1]) for score in scores][0]
