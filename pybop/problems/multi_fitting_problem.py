from typing import Optional

import numpy as np

from pybop import BaseProblem, Dataset
from pybop.parameters.parameter import Inputs, Parameters


class MultiFittingProblem(BaseProblem):
    """
    Problem class for joining mulitple fitting problems into one combined fitting problem.

    Extends `BaseProblem` in a similar way to FittingProblem but for multiple parameter
    estimation problems, which must first be defined individually.

    Additional Attributes
    ---------------------
    problems : pybop.FittingProblem
        The individual PyBOP fitting problems.
    """

    def __init__(self, *args):
        self.problems = []
        models_to_check = []
        for problem in args:
            self.problems.append(problem)
            if problem.model is not None:
                models_to_check.append(problem.model)

        # Check that there are no copies of the same model
        if len(set(models_to_check)) < len(models_to_check):
            raise ValueError("Make a new_copy of the model for each problem.")

        # Compile the set of parameters, ignoring duplicates
        combined_parameters = Parameters()
        for problem in self.problems:
            combined_parameters.join(problem.parameters)

        # Combine the target datasets
        domain = self.problems[0].domain
        combined_domain_data = []
        combined_signal = []
        for problem in self.problems:
            domain = problem.domain if problem.domain == domain else "Mixed domain"
            for signal in problem.signal:
                combined_domain_data.extend(problem.domain_data)
                combined_signal.extend(problem.target[signal])

        super().__init__(
            parameters=combined_parameters,
            model=None,
            signal=["Combined signal"],
        )

        self.domain = domain
        combined_dataset = Dataset(
            {
                self.domain: np.asarray(combined_domain_data),
                "Combined signal": np.asarray(combined_signal),
            }
        )
        self._dataset = combined_dataset.data
        self.parameters.initial_value()

        # Unpack domain and target data
        self._domain_data = self._dataset[self.domain]
        self.n_domain_data = len(self._domain_data)
        self.set_target(combined_dataset)

    def set_initial_state(self, initial_state: Optional[dict] = None):
        """
        Set the initial state to be applied to evaluations of the problem.

        Parameters
        ----------
        initial_state : dict, optional
            A valid initial state (default: None).
        """
        for problem in self.problems:
            problem.set_initial_state(initial_state)

    def evaluate(self, inputs: Inputs):
        """
        Evaluate the model with the given parameters and return the signal.

        Parameters
        ----------
        inputs : Inputs
            Parameters for evaluation of the model.

        Returns
        -------
        y : np.ndarray
            The model output y(t) simulated with given inputs.
        """
        inputs = self.parameters.verify(inputs)
        self.parameters.update(values=list(inputs.values()))

        combined_domain = []
        combined_signal = []

        for problem in self.problems:
            problem_inputs = problem.parameters.as_dict()
            problem_output = problem.evaluate(problem_inputs)
            domain_data = (
                problem_output[problem.domain]
                if problem.domain in problem_output.keys()
                else problem.domain_data[: len(problem_output[problem.signal[0]])]
            )

            # Collect signals
            for signal in problem.signal:
                combined_domain.extend(domain_data)
                combined_signal.extend(problem_output[signal])

        return {
            self.domain: np.asarray(combined_domain),
            "Combined signal": np.asarray(combined_signal),
        }

    def evaluateS1(self, inputs: Inputs):
        """
        Evaluate the model with the given parameters and return the signal and its derivatives.

        Parameters
        ----------
        inputs : Inputs
            Parameters for evaluation of the model.

        Returns
        -------
        tuple[dict, np.ndarray]
            A tuple containing the simulation result y(t) as a dictionary and the sensitivities
            dy/dx(t) evaluated with given inputs.
        """
        inputs = self.parameters.verify(inputs)
        self.parameters.update(values=list(inputs.values()))

        combined_domain = []
        combined_signal = []
        dy = dict.fromkeys(self.parameters.keys())
        for key in self.parameters.keys():
            dy[key] = {"Combined signal": []}

        for problem in self.problems:
            problem_inputs = problem.parameters.as_dict()
            problem_output, dyi = problem.evaluateS1(problem_inputs)
            domain_data = (
                problem_output[problem.domain]
                if problem.domain in problem_output.keys()
                else problem.domain_data[: len(problem_output[problem.signal[0]])]
            )

            # Collect signals and derivatives
            for signal in problem.signal:
                combined_domain.extend(domain_data)
                combined_signal.extend(problem_output[signal])
                for key in self.parameters.keys():
                    dy[key]["Combined signal"].extend(dyi[key][signal])

        y = {
            self.domain: np.asarray(combined_domain),
            "Combined signal": np.asarray(combined_signal),
        }

        return (y, dy)

    @property
    def pybamm_solution(self):
        solution_list = []
        for problem in self.problems:
            solution_list.append(problem.pybamm_solution)
        return solution_list if any(solution_list) else None
