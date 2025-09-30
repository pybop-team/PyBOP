import numpy as np

from pybop import Problem
from pybop.parameters.parameter import Inputs, Parameters


class MetaProblem(Problem):
    """
    Problem class for joining mulitple problems into one combined problem.

    Evaluates multiple problems, which must first be defined individually.

    Parameters
    ----------
    problems : pybop.Problem
        The individual PyBOP fitting problems.
    """

    def __init__(self, *problems, weights: list[float] | None = None):
        if not all(isinstance(problem, Problem) for problem in problems):
            raise TypeError("All problems must be instances of Problem.")
        self.problems = [problem for problem in problems]

        # Compile the set of parameters, ignoring duplicates
        combined_parameters = Parameters()
        sensitivities_available = True
        for problem in self.problems:
            combined_parameters.join(problem.parameters)
            if not problem.has_sensitivities:
                sensitivities_available = False

        super().__init__(simulator=None, cost=None)
        self.parameters = combined_parameters
        self._has_sensitivities = sensitivities_available

        # Check if weights are provided
        if weights is not None:
            try:
                self.weights = np.asarray(weights, dtype=float)
            except ValueError:
                raise ValueError("Weights must be numeric values.") from None

            if self.weights.size != len(self.problems):
                raise ValueError("Number of weights must match number of problems.")
        else:
            self.weights = np.ones(len(self.problems))

        # Apply the minimising property from each problem
        for i, problem in enumerate(self.problems):
            self.weights[i] = self.weights[i] * (1 if problem.minimising else -1)
        if all(not problem.minimising for problem in self.problems):
            # If all problems are maximising, convert the weighted problem to maximising
            self.weights = -self.weights
            self._minimising = False

    def single_call(
        self,
        inputs: Inputs,
        calculate_grad: bool,
    ) -> float | tuple[float, np.ndarray]:
        """Evaluate the problem and (optionally) the gradient for a single set of inputs."""
        e = np.empty_like(self.problems)
        de = np.empty((len(self.parameters), len(self.problems)))

        for i, problem in enumerate(self.problems):
            if calculate_grad:
                e[i], de[:, i] = problem.single_call(
                    inputs, calculate_grad=calculate_grad
                )
            else:
                e[i] = problem.single_call(inputs, calculate_grad=calculate_grad)

        e = np.dot(e, self.weights)
        if calculate_grad:
            de = np.dot(de, self.weights)
            return e, de

        return e
