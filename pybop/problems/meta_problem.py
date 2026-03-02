import numpy as np

from pybop.costs.evaluation import Evaluation
from pybop.parameters.parameter import Inputs, Parameters
from pybop.problems.problem import Problem


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
        self._parameters = combined_parameters
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

    def get_problem_inputs(self, inputs: Inputs, i: int):
        return {key: inputs[key] for key in self.problems[i].parameters.keys()}

    def evaluate_batch(
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
        Evaluation
            Cost values of len(inputs) and (optionally) the gradient of the cost with respect to
            each input parameter with shape (len(inputs), len(parameters)).
        """
        n_inputs = len(inputs)
        n_problems = len(self.problems)
        e = np.empty((n_inputs, n_problems))
        de = {key: np.zeros((n_inputs, n_problems)) for key in self.parameters.keys()}

        for i, problem in enumerate(self.problems):
            problem_inputs = [self.get_problem_inputs(x, i) for x in inputs]
            if calculate_sensitivities:
                e[:, i], sensitivities = problem.evaluate_batch(
                    problem_inputs, calculate_sensitivities=calculate_sensitivities
                ).get_values()
                for key, value in sensitivities.items():
                    de[key][:, i] = value
            else:
                e[:, i] = problem.evaluate_batch(
                    problem_inputs, calculate_sensitivities=calculate_sensitivities
                ).values

        e = np.dot(e, self.weights)
        if calculate_sensitivities:
            for key in de.keys():
                de[key] = np.dot(de[key], self.weights)
            return Evaluation(values=e, sensitivities=de)

        return Evaluation(values=e)

    def set_target(self, value: list[str] | str | None = None):
        for problem in self.problems:
            problem.set_target(value)

    @property
    def has_sensitivities(self):
        if all(problem.has_sensitivities for problem in self.problems):
            return True
        return False
