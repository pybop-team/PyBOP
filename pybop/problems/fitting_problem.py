import numpy as np

from pybop import BaseProblem
from pybop._dataset import Dataset
from pybop.parameters.parameter import Inputs, Parameters


class FittingProblem(BaseProblem):
    """
    Problem class for fitting (parameter estimation) problems.

    Extends `BaseProblem` with specifics for fitting a model to a dataset.

    Parameters
    ----------
    model : object
        The model to fit.
    parameters : pybop.Parameter or pybop.Parameters
        An object or list of the parameters for the problem.
    dataset : Dataset
        Dataset object containing the data to fit the model to.
    signal : str, optional
        The variable used for fitting (default: "Voltage [V]").
    additional_variables : List[str], optional
        Additional variables to observe and store in the solution (default additions are: ["Time [s]"]).
    init_soc : float, optional
        Initial state of charge (default: None).
    """

    def __init__(
        self,
        model,
        parameters,
        dataset,
        check_model=True,
        signal=None,
        additional_variables=None,
        init_soc=None,
    ):
        # Add time and remove duplicates
        if additional_variables is None:
            additional_variables = []
        if signal is None:
            signal = ["Voltage [V]"]
        additional_variables.extend(["Time [s]"])
        additional_variables = list(set(additional_variables))

        super().__init__(
            parameters, model, check_model, signal, additional_variables, init_soc
        )
        self._dataset = dataset.data
        self.parameters.initial_value()

        # Check that the dataset contains time and current
        dataset.check([*self.signal, "Current function [A]"])

        # Unpack time and target data
        self._time_data = self._dataset["Time [s]"]
        self.n_time_data = len(self._time_data)
        self.set_target(dataset)

        # Add useful parameters to model
        if model is not None:
            self._model.signal = self.signal
            self._model.additional_variables = self.additional_variables
            self._model.n_outputs = self.n_outputs
            self._model.n_time_data = self.n_time_data

            # Build the model from scratch
            if self._model._built_model is not None:
                self._model._model_with_set_params = None
                self._model._built_model = None
                self._model._built_initial_soc = None
                self._model._mesh = None
                self._model._disc = None
            self._model.build(
                dataset=self._dataset,
                parameters=self.parameters,
                check_model=self.check_model,
                init_soc=self.init_soc,
            )

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

        requires_rebuild = False
        for key, value in inputs.items():
            if key in self._model.rebuild_parameters:
                current_value = self.parameters[key].value
                if value != current_value:
                    self.parameters[key].update(value=value)
                    requires_rebuild = True

        if requires_rebuild:
            self._model.rebuild(parameters=self.parameters)

        y = self._model.simulate(inputs=inputs, t_eval=self._time_data)

        return y

    def evaluateS1(self, inputs: Inputs):
        """
        Evaluate the model with the given parameters and return the signal and its derivatives.

        Parameters
        ----------
        inputs : Inputs
            Parameters for evaluation of the model.

        Returns
        -------
        tuple
            A tuple containing the simulation result y(t) and the sensitivities dy/dx(t) evaluated
            with given inputs.
        """
        inputs = self.parameters.verify(inputs)

        if self._model.rebuild_parameters:
            raise RuntimeError(
                "Gradient not available when using geometric parameters."
            )

        y, dy = self._model.simulateS1(
            inputs=inputs,
            t_eval=self._time_data,
        )

        return (y, np.asarray(dy))


class MultiFittingProblem(BaseProblem):
    """
    Problem class for joining mulitple fitting problems.

    Extends `FittingProblem` to multiple fitting problems.
    """

    def __init__(self, problem_list):
        self.problem_list = problem_list

        # Compile the set of parameters, ignoring duplicates
        combined_parameters = Parameters()
        for problem in self.problem_list:
            combined_parameters.join(problem.parameters)

        # Combine the target datasets
        combined_dataset = Dataset(
            {"Time [s]": np.asarray([]), "Combined signal": np.asarray([])}
        )
        for problem in self.problem_list:
            for signal in problem.signal:
                combined_dataset["Time [s]"] = np.concatenate(
                    (combined_dataset["Time [s]"], problem._time_data)
                )
                combined_dataset["Combined signal"] = np.concatenate(
                    (combined_dataset["Combined signal"], problem._target[signal])
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

        y = {"Combined signal": np.asarray([])}
        for problem in self.problem_list:
            problem_inputs = problem.parameters.as_dict()
            for signal in problem.signal:
                yi = problem.evaluate(problem_inputs)
                y["Combined signal"] = np.concatenate(
                    (y["Combined signal"], yi[signal])
                )

        return y

    def evaluateS1(self, inputs: Inputs):
        """
        Evaluate the model with the given parameters and return the signal and its derivatives.

        Parameters
        ----------
        inputs : Inputs
            Parameters for evaluation of the model.

        Returns
        -------
        tuple
            A tuple containing the simulation result y(t) and the sensitivities dy/dx(t) evaluated
            with given inputs.
        """

        inputs = self.parameters.verify(inputs)
        self.parameters.update(values=list(inputs.values()))

        # y = np.empty((self._target_length))
        # dy = np.empty((self._target_length, self.n_parameters))

        y = {"Combined signal": np.asarray([])}
        dy = None
        for problem in self.problem_list:
            problem_inputs = problem.parameters.as_dict()
            for signal in problem.signal:
                yi, dyi = problem.evaluateS1(problem_inputs)
                y["Combined signal"] = np.concatenate(
                    (y["Combined signal"], yi[signal])
                )
                if dy is None:
                    dy = dyi
                else:
                    dy = np.concatenate((dy, dyi))

        return (y, dy)
