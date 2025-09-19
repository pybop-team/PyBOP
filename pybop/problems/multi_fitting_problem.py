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
        sims_to_check = []
        for problem in args:
            self.problems.append(problem)
            if problem.simulator is not None:
                sims_to_check.append(problem.simulator)

        # Check that there are no copies of the same model
        if len(set(sims_to_check)) < len(sims_to_check):
            raise ValueError("Make a new_copy of the simulator for each problem.")

        # Compile the set of parameters, ignoring duplicates
        combined_parameters = Parameters()
        sensitivities_available = True
        for problem in self.problems:
            combined_parameters.join(problem.parameters)
            if not problem.has_sensitivities:
                sensitivities_available = False

        # Combine the target datasets
        domain = self.problems[0].domain
        combined_domain_data = []
        combined_output_variables = []
        for problem in self.problems:
            domain = problem.domain if problem.domain == domain else "Mixed domain"
            for var in problem.output_variables:
                combined_domain_data.extend(problem.domain_data)
                combined_output_variables.extend(problem.target[var])

        super().__init__(
            simulator=None,
            parameters=combined_parameters,
            output_variables=["Combined signal"],
        )

        self.domain = domain
        combined_dataset = Dataset(
            {
                self.domain: np.asarray(combined_domain_data),
                "Combined signal": np.asarray(combined_output_variables),
            }
        )
        self._dataset = combined_dataset.data
        self.parameters.get_initial_values()
        self._has_sensitivities = sensitivities_available

        # Unpack domain and target data
        self._domain_data = self._dataset[self.domain]
        self.n_domain_data = len(self._domain_data)
        self.set_target(combined_dataset)

    def evaluate(self, inputs: Inputs | list[Inputs]):
        """
        Evaluate the model with the given parameters and return the output variables.

        Parameters
        ----------
        inputs : Inputs | list[Inputs]
            Parameters for evaluation of the model.

        Returns
        -------
        y : np.ndarray
            The model output y(t) simulated with given inputs.
        """
        combined_domain = []
        combined_output_variables = []

        for problem in self.problems:
            problem_inputs = problem.parameters.to_dict()
            problem_output = problem.evaluate(problem_inputs)
            domain_data = (
                problem_output[problem.domain]
                if problem.domain in problem_output.keys()
                else problem.domain_data[
                    : len(problem_output[problem.output_variables[0]])
                ]
            )

            # Collect output variables
            for var in problem.output_variables:
                combined_domain.extend(domain_data)
                combined_output_variables.extend(problem_output[var])

        return {
            self.domain: np.asarray(combined_domain),
            "Combined signal": np.asarray(combined_output_variables),
        }

    def evaluateS1(self, inputs: Inputs):
        """
        Evaluate the model with the given parameters and return the output_variables and
        their derivatives.

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
        self.parameters.update(values=list(inputs.values()))

        combined_domain = []
        combined_output_variables = []
        dy = dict.fromkeys(self.parameters.keys())
        for key in self.parameters.keys():
            dy[key] = {"Combined signal": []}

        for problem in self.problems:
            problem_inputs = problem.parameters.to_dict()
            problem_output, dyi = problem.evaluateS1(problem_inputs)
            domain_data = (
                problem_output[problem.domain]
                if problem.domain in problem_output.keys()
                else problem.domain_data[
                    : len(problem_output[problem.output_variables[0]])
                ]
            )

            # Collect output variables and derivatives
            for var in problem.output_variables:
                combined_domain.extend(domain_data)
                combined_output_variables.extend(problem_output[var])
                for key in self.parameters.keys():
                    dy[key]["Combined signal"].extend(dyi[key][var])

        y = {
            self.domain: np.asarray(combined_domain),
            "Combined signal": np.asarray(combined_output_variables),
        }

        return (y, dy)
