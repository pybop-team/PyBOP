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
        for problem in args:
            if problem.model is not None:
                # Take a copy of the model and build from scratch
                problem._model = problem.model.new_copy()
                problem.model.build(
                    dataset=problem.dataset,
                    parameters=problem.parameters,
                    check_model=problem.check_model,
                    initial_state=problem.initial_state,
                )
            self.problems.append(problem)

        # Compile the set of parameters, ignoring duplicates
        combined_parameters = Parameters()
        for problem in self.problems:
            combined_parameters.join(problem.parameters)

        # Combine the target datasets
        combined_time_data = []
        combined_signal = []
        for problem in self.problems:
            for signal in problem.signal:
                combined_time_data.extend(problem.time_data)
                combined_signal.extend(problem.target[signal])
        combined_dataset = Dataset(
            {
                "Time [s]": np.asarray(combined_time_data),
                "Combined signal": np.asarray(combined_signal),
            }
        )

        super().__init__(
            parameters=combined_parameters,
            model=None,
            signal=["Combined signal"],
        )
        self._dataset = combined_dataset.data
        self.parameters.initial_value()

        # Unpack time and target data
        self._time_data = self._dataset["Time [s]"]
        self.n_time_data = len(self._time_data)
        self.set_target(combined_dataset)

    def evaluate(self, inputs: Inputs, **kwargs):
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

        combined_signal = []

        for problem in self.problems:
            problem_inputs = problem.parameters.as_dict()
            signal_values = problem.evaluate(problem_inputs, **kwargs)

            # Collect signals
            for signal in problem.signal:
                combined_signal.extend(signal_values[signal])

        return {"Combined signal": np.asarray(combined_signal)}

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

        combined_signal = []
        all_derivatives = []

        for problem in self.problems:
            problem_inputs = problem.parameters.as_dict()
            signal_values, dyi = problem.evaluateS1(problem_inputs)

            # Collect signals and derivatives
            for signal in problem.signal:
                combined_signal.extend(signal_values[signal])
            all_derivatives.append(dyi)

        y = {"Combined signal": np.asarray(combined_signal)}
        dy = np.concatenate(all_derivatives) if all_derivatives else None

        return (y, dy)
