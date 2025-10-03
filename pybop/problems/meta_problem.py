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

    def batch_evaluate(
        self,
        inputs: list[Inputs],
        calculate_sensitivities: bool,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Evaluate each problem for each set of inputs and return the cost values and (optionally)
        the sensitivities with respect to each input parameter.

        Parameters
        ----------
        inputs : list[Inputs]
            A list of input parameters.
        calculate_sensitivities : bool
            Whether to also return the sensitivities (default: False).

        Returns
        -------
        np.ndarray | CostsAndSensitivities
            Cost values of len(inputs) and (optionally) the gradient of the cost with respect to
            each input parameter with shape (len(inputs), len(parameters)).
        """
        n_inputs = len(inputs)
        n_problems = len(self.problems)
        e = np.empty((n_inputs, n_problems))
        de = np.empty((n_inputs, len(self.parameters), n_problems))

        for i, problem in enumerate(self.problems):
            if calculate_sensitivities:
                e[:, i], de[:, :, i] = problem.batch_evaluate(
                    inputs, calculate_sensitivities=calculate_sensitivities
                )
            else:
                e[:, i] = problem.batch_evaluate(
                    inputs, calculate_sensitivities=calculate_sensitivities
                )

        e = np.dot(e, self.weights)
        if calculate_sensitivities:
            de = np.dot(de, self.weights)
            return e, de

        return e
