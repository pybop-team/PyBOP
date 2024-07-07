import numpy as np

from pybop import BaseProblem
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
    additional_variables : List[str], optional
        Additional variables to observe and store in the solution (default additions are: ["Time [s]", "Current [A]"]).
    init_soc : float, optional
        Initial state of charge (default: None).
    """

    def __init__(
        self,
        model,
        parameters,
        experiment,
        check_model=True,
        signal=None,
        additional_variables=None,
        init_soc=None,
    ):
        # Add time and current and remove duplicates
        if additional_variables is None:
            additional_variables = []
        if signal is None:
            signal = ["Voltage [V]"]
        additional_variables.extend(["Time [s]", "Current [A]"])
        additional_variables = list(set(additional_variables))

        super().__init__(
            parameters,
            model,
            check_model,
            signal,
            additional_variables,
            init_soc,
        )
        self.experiment = experiment

        # Build the model if required
        if experiment is not None:
            # Leave the build until later to apply the experiment
            self._model.classify_and_update_parameters(self.parameters)

        elif self._model._built_model is None:
            self._model.build(
                experiment=self.experiment,
                parameters=self.parameters,
                check_model=self.check_model,
                init_soc=self.init_soc,
            )

        # Add an example dataset for plotting comparison
        sol = self.evaluate(self.parameters.as_dict("initial"))
        self._time_data = sol["Time [s]"]
        self._target = {key: sol[key] for key in self.signal}
        self._dataset = None

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
            The model output y(t) simulated with inputs.
        """
        inputs = self.parameters.verify(inputs)

        sol = self._model.predict(
            inputs=inputs,
            experiment=self.experiment,
            init_soc=self.init_soc,
        )

        if sol == [np.inf]:
            return sol

        else:
            predictions = {}
            for signal in self.signal + self.additional_variables:
                predictions[signal] = sol[signal].data

        return predictions
