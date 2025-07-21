import numpy as np
from pints import Evaluator as PintsEvaluator


class ScipyEvaluator(PintsEvaluator):
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

        return [(score[0], score[1]) for score in scores][0]
