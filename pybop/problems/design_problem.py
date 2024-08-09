import warnings
from typing import Optional

import numpy as np

from pybop import BaseModel, BaseProblem, Experiment, Parameters
from pybop.models.empirical.base_ecm import ECircuitModel
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
    initial_state : dict, optional
        A valid initial state (default: {"Initial SoC": 1.0}).
    """

    def __init__(
        self,
        model: BaseModel,
        parameters: Parameters,
        experiment: Optional[Experiment],
        check_model: bool = True,
        signal: Optional[list[str]] = None,
        additional_variables: Optional[list[str]] = None,
        initial_state: Optional[dict] = None,
    ):
        # Add time and current and remove duplicates
        additional_variables = additional_variables or []
        additional_variables.extend(["Time [s]", "Current [A]"])
        additional_variables = list(set(additional_variables))

        super().__init__(
            parameters, model, check_model, signal, additional_variables, initial_state
        )
        self.experiment = experiment

        # Add an example dataset for plotting comparison
        sol = self.evaluate(self.parameters.as_dict("initial"))
        self._time_data = sol["Time [s]"]
        self._target = {key: sol[key] for key in self.signal}
        self._dataset = None
        self.warning_patterns = [
            "Ah is greater than",
            "Non-physical point encountered",
        ]

    def set_initial_state(self, initial_state):
        """
        Set the initial state to be applied to evaluations of the problem.

        Parameters
        ----------
        initial_state : dict, optional
            A valid initial state (default: None).
        """
        if initial_state is None:
            if isinstance(self.model, ECircuitModel):
                initial_state = {"Initial SoC": self.model.parameter_set["Initial SoC"]}
            else:
                initial_state = {"Initial SoC": 1.0}  # default value
        elif "Initial open-circuit voltage [V]" in initial_state.keys():
            warnings.warn(
                "It is usually better to define an initial state of charge as the "
                "initial_state for a DesignProblem because this state will scale with "
                "design properties such as the capacity of the battery, as opposed to the "
                "initial open-circuit voltage which may correspond to a different state "
                "of charge for each design.",
                UserWarning,
                stacklevel=1,
            )

        self.initial_state = initial_state

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
                    initial_state=self.initial_state,
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
