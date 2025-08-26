import numpy as np
from pints import Evaluator as PintsEvaluator

from pybop.problems.base_problem import Problem


class BaseEvaluator(PintsEvaluator):
    """
    Evaluates a function (or callable object) for multiple positions.

    Applies transformations, if any.

    Based on and extends Pints' :class:`Evaluator`.

    Parameters
    ----------
    problem : pybop.Problem
        The problem to be optimised.
    minimise : bool
        If True, the cost function is minimised, otherwise maximisation is performed by
        inverting the sign of the cost function and sensitivities.
    with_sensitivities : bool
        If True, the sensitivities are calculated as well as the cost.
    """

    def __init__(self, problem: Problem, minimise: bool, with_sensitivities: bool):
        self.transformation = problem.params.transformation

        # Choose which function to evaluate
        if minimise and with_sensitivities:

            def fun(x):
                problem.set_params(x)
                cost, grad = problem.run_with_sensitivities()
                jac = self.transformation.jacobian(x)
                grad = np.matmul(grad, jac)
                return cost, grad

        elif minimise:

            def fun(x):
                problem.set_params(x)
                return problem.run()

        elif not minimise and with_sensitivities:

            def fun(x):
                problem.set_params(x)
                loss, grad = problem.run_with_sensitivities()

                # Transform the gradient, for multiple parameters
                # iterate across each and apply the transformation.
                if grad.ndim == 1:
                    jac = self.transformation.jacobian(x)
                    grad = np.dot(grad, jac)
                else:
                    for i in range(grad.shape[0]):
                        jac = self.transformation.jacobian(x[i])
                        grad[i] = np.dot(grad[i], jac)

                return (-loss, -grad)

        else:

            def fun(x):
                problem.set_params(x)
                return -problem.run()

        # Pass function to PintsEvaluator
        super().__init__(fun)


class PopulationEvaluator(BaseEvaluator):
    """
    Evaluates a function (or callable object) for multiple positions.

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


class SequentialEvaluator(BaseEvaluator):
    """
    Evaluates a function (or callable object) for a list of input values, and
    returns a list containing the calculated function evaluations.

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


class SciPyEvaluator(BaseEvaluator):
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
