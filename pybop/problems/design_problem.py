import warnings
from typing import Optional

import numpy as np

from pybop import BaseModel, BaseProblem, Experiment, Parameters
from pybop.parameters.parameter import Inputs


class DesignProblem(BaseProblem):
    """
    Problem class for design optimization problems.

    Extends `BaseProblem` with specifics for applying a model to an experimental design.

    Parameters
    ----------
    model : object
        The model to apply the design to.
    parameters : pybop.Parameter or pybop.Parameters
        An object or list of the parameters for the problem.
    experiment : object
        The experimental setup to apply the model to.
    check_model : bool, optional
        Flag to indicate if the model parameters should be checked for feasibility each iteration (default: True).
    signal : str, optional
        The signal to fit (default: "Voltage [V]").
    additional_variables : list[str], optional
        Additional variables to observe and store in the solution (default additions are: ["Time [s]", "Current [A]"]).
    init_soc : float, optional
        Initial state of charge (default: 1.0).
    """

    def __init__(
        self,
        model: BaseModel,
        parameters: Parameters,
        experiment: Optional[Experiment],
        check_model: bool = True,
        signal: Optional[list[str]] = None,
        additional_variables: Optional[list[str]] = None,
        init_soc=None,
    ):
        # Add time and current and remove duplicates
        additional_variables = additional_variables or []
        additional_variables.extend(["Time [s]", "Current [A]"])
        additional_variables = list(set(additional_variables))

        if init_soc is None:
            if "Initial SoC" in model._parameter_set.keys():
                init_soc = model._parameter_set["Initial SoC"]
            else:
                init_soc = 1.0  # default value

        super().__init__(parameters, model, check_model, signal, additional_variables)
        self.experiment = experiment
        self.init_soc = init_soc
        self.parameters.initial_value()

        # Add an example dataset for plotting comparison
        sol = self.evaluate(self.parameters.as_dict("initial"))
        self._time_data = sol["Time [s]"]
        self._target = {key: sol[key] for key in self.signal}
        self._dataset = None
        self.warning_patterns = [
            "Ah is greater than",
            "Non-physical point encountered",
        ]

    def evaluate(self, inputs: Inputs, update_capacity=False):
        """
        Evaluate the model with the given parameters and return the signal.

        Parameters
        ----------
        inputs : Inputs
            Parameters for evaluation of the model.

        Returns
        -------
        y : np.ndarray
            The model output y(t) simulated with inputs.
        """
        inputs = self.parameters.verify(inputs)
        if update_capacity:
            self.model.approximate_capacity(inputs)

        try:
            with warnings.catch_warnings():
                for pattern in self.warning_patterns:
                    warnings.filterwarnings(
                        "error", category=UserWarning, message=pattern
                    )

                sol = self._model.predict(
                    inputs=inputs,
                    experiment=self.experiment,
                    initial_state=self.init_soc,
                )

        # Catch infeasible solutions and return infinity
        except (UserWarning, Exception) as e:
            if self.verbose:
                print(f"Ignoring this sample due to: {e}")
            return {
                signal: np.asarray(np.ones(2) * -np.inf)
                for signal in [*self.signal, *self.additional_variables]
            }

        return {
            signal: sol[signal].data
            for signal in self.signal + self.additional_variables
        }
