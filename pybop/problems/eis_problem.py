import warnings
from typing import Optional

from pybop import BaseModel, BaseProblem, Dataset, Parameters
from pybop.parameters.parameter import Inputs


class EISProblem(BaseProblem):
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
    init_state : float, optional
        Initial state of charge (default: None).
    """

    def __init__(
        self,
        model: BaseModel,
        parameters: Parameters,
        dataset: Dataset,
        check_model: bool = True,
        signal: Optional[list[str]] = None,
        additional_variables: Optional[list[str]] = None,
        initial_state: Optional[dict] = None,
    ):
        # Add frequency and remove duplicates
        additional_variables = additional_variables or []
        additional_variables.extend(["Frequency [Hz]"])
        additional_variables = list(set(additional_variables))

        super().__init__(
            parameters, model, check_model, signal, additional_variables, initial_state
        )
        self._dataset = dataset.data
        self._n_parameters = len(self.parameters)

        # Check that the dataset contains necessary variables
        dataset.check(
            domain="Frequency [Hz]", signal=[*self.signal, "Current function [A]"]
        )

        # Unpack time and target data
        self._domain_data = self._dataset["Frequency [Hz]"]
        self.n_data = len(self._domain_data)
        self.set_target(dataset)

        if model is not None:
            # Build the model from scratch
            if self._model.built_model is not None:
                self._model.clear()
            self._model.build(
                dataset=self._dataset,
                parameters=self.parameters,
                check_model=self.check_model,
                initial_state=self.initial_state,
            )

    def set_initial_state(self, initial_state: Optional[dict] = None):
        """
        Set the initial state to be applied to evaluations of the problem.

        Parameters
        ----------
        initial_state : dict, optional
            A valid initial state (default: None).
        """
        if initial_state is not None and "Initial SoC" in initial_state.keys():
            warnings.warn(
                "It is usually better to define an initial open-circuit voltage as the "
                "initial_state for a FittingProblem because this value can typically be "
                "obtained from the data, unlike the intrinsic initial state of charge. "
                "In the case where the fitting parameters do not change the OCV-SOC "
                "relationship, the initial state of charge may be passed to the model "
                'using, for example, `model.set_initial_state({"Initial SoC": 1.0})` '
                "before constructing the FittingProblem.",
                UserWarning,
                stacklevel=1,
            )

        self.initial_state = initial_state

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

        return self._model.simulateEIS(
            inputs=inputs,
            f_eval=self._domain_data,
            # initial_state=self.initial_state,
        )

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
        return NotImplementedError
