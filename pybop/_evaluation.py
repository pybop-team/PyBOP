import numpy as np
from pints import Evaluator as PintsEvaluator


class PopulationEvaluator(PintsEvaluator):
    """
    Evaluates a function (or callable object)
    for multiple positions.

    Parameters
    ----------
    function : callable
        The function to evaluate. This function should accept a list-like
        object of positions to be evaluated.
    args : sequence, optional
        A sequence containing extra arguments to be passed to the function.
        If specified, the function will be called as `function(x, *args)`.
    """

    def _evaluate(self, positions):
        return self._function(positions, *self._args)


class SequentialEvaluator(PintsEvaluator):
    """
    Evaluates a function (or callable object) for a list of input values, and
    returns a list containing the calculated function evaluations.

    Based on and extends Pints' :class:`Evaluator`.

    Parameters
    ----------
    function : callable
        The function to evaluate.
    args : sequence
        An optional tuple containing extra arguments to ``f``. If ``args`` is
        specified, ``f`` will be called as ``f(x, *args)``.
    """

    def _evaluate(self, positions):
        scores = [self._function(x, *self._args) for x in positions]

        # Non-sensitivity costs should be a singular dimension array
        if not isinstance(scores[0], tuple):
            return np.asarray(scores).reshape(-1)

        return scores


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
            return np.asarray(scores).reshape(-1)

        return [(score[0], score[1]) for score in scores][0]
