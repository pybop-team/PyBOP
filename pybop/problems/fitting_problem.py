import warnings
from typing import Optional

import numpy as np

from pybop import BaseModel, BaseProblem, Dataset
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
    check_model : bool, optional
        Flag to indicate if the model should be checked (default: True).
    signal : str, optional
        The variable used for fitting (default: "Voltage [V]").
    additional_variables : list[str], optional
        Additional variables to observe and store in the solution (default additions are: ["Time [s]"]).
    initial_state : dict, optional
        A valid initial state, e.g. the initial open-circuit voltage (default: None).

    Additional Attributes
    ---------------------
    dataset : dictionary
        The dictionary from a Dataset object containing the signal keys and values to fit the model to.
    domain_data : np.ndarray
        The domain points in the dataset.
    n_domain_data : int
        The number of domain points.
    target : np.ndarray
        The target values of the signals.
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
        super().__init__(
            parameters, model, check_model, signal, additional_variables, initial_state
        )
        self._dataset = dataset.data
        self._n_parameters = len(self.parameters)

        # Check that the dataset contains necessary variables
        dataset.check(domain=self.domain, signal=[*self.signal, "Current function [A]"])

        # Unpack domain and target data
        self._domain_data = self._dataset[self.domain]
        self.n_data = len(self._domain_data)
        self.set_target(dataset)

        if self._model is not None:
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

    def evaluate(
        self,
        inputs: Inputs,
        update_capacity=False,
    ) -> dict[str, np.ndarray[np.float64]]:
        """
        Evaluate the model with the given parameters and return the signal.

        Parameters
        ----------
        inputs : Inputs
            Parameters for evaluation of the model.

        Returns
        -------
        y : np.ndarray
            The simulated model output y(t) for self.eis == False, and y(ω) for self.eis == True for the given inputs.
        """
        inputs = self.parameters.verify(inputs)
        if self.eis:
            return self._evaluateEIS(inputs, update_capacity=update_capacity)
        else:
            try:
                sol = self._model.simulate(
                    inputs=inputs,
                    t_eval=self._domain_data,
                    initial_state=self.initial_state,
                )
            except Exception as e:
                if self.verbose:
                    print(f"Simulation error: {e}")
                return {signal: self.failure_output for signal in self.signal}

            return {
                signal: sol[signal].data
                for signal in (self.signal + self.additional_variables)
            }

    def _evaluateEIS(
        self, inputs: Inputs, update_capacity=False
    ) -> dict[str, np.ndarray[np.float64]]:
        """
        Evaluate the model with the given parameters and return the signal.

        Parameters
        ----------
        inputs : Inputs
            Parameters for evaluation of the model.

        Returns
        -------
        y : np.ndarray
            The simulated model output y(ω) for the given inputs.
        """
        try:
            sol = self._model.simulateEIS(
                inputs=inputs,
                f_eval=self._domain_data,
                initial_state=self.initial_state,
            )
        except Exception as e:
            if self.verbose:
                print(f"Simulation error: {e}")
            return {signal: self.failure_output for signal in self.signal}

        return sol

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

        try:
            sol = self._model.simulateS1(
                inputs=inputs,
                t_eval=self._domain_data,
                initial_state=self.initial_state,
            )
        except Exception as e:
            print(f"Error: {e}")
            return {
                signal: self.failure_output for signal in self.signal
            }, self.failure_output

        y = {signal: sol[signal].data for signal in self.signal}

        # Extract the sensitivities and stack them along a new axis for each signal
        dy = np.empty(
            (
                sol[self.signal[0]].data.shape[0],
                self.n_outputs,
                self._n_parameters,
            )
        )

        for i, signal in enumerate(self.signal):
            dy[:, i, :] = np.stack(
                [
                    sol[signal].sensitivities[key].toarray()[:, 0]
                    for key in self.parameters.keys()
                ],
                axis=-1,
            )

        return y, np.asarray(dy)
