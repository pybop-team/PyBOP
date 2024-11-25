import numpy as np
from pints import Evaluator as PintsEvaluator


class SequentialJaxEvaluator(PintsEvaluator):
    """
    Evaluates a JAX function (or callable object) for a list of input values,
    and returns a list or NumPy array of the calculated function evaluations.

    This class extends PintsEvaluator to provide an interface for evaluating
    functions using JAX, with optional additional arguments passed to the
    function.

    Parameters
    ----------
    function : callable
        The function to evaluate. This function should accept an input and
        optionally additional arguments, returning either a single value or a tuple.
    args : sequence, optional
        A sequence containing extra arguments to be passed to the function.
        If specified, the function will be called as `function(x, *args)`.
    """

    def __init__(self, function, args=None):
        super().__init__(function, args)

    def _evaluate(self, positions):
        scores = [self._function(x, *self._args) for x in positions]

        # If gradient provided, convert jnp to np and return
        if isinstance(scores[0], tuple):
            return [(np.asarray(score[0]), score[1]) for score in scores]

        return np.asarray(scores)
