import warnings
from typing import Optional

import numpy as np
from pybamm import IDAKLUJax, SolverError

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
    signal : list[str], optional
        A list of variables to fit (default: ["Voltage [V]"]).
    domain : str, optional
        The name of the domain (default: "Time [s]").
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
        domain: Optional[str] = None,
        additional_variables: Optional[list[str]] = None,
        initial_state: Optional[dict] = None,
    ):
        super().__init__(
            parameters=parameters,
            model=model,
            check_model=check_model,
            signal=signal,
            domain=domain,
            additional_variables=additional_variables,
            initial_state=initial_state,
        )
        self._dataset = dataset.data
        self._n_parameters = len(self.parameters)

        # Check that the dataset contains necessary variables
        dataset.check(domain=self.domain, signal=self.signal)

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

    def evaluate(self, inputs: Inputs) -> dict[str, np.ndarray[np.float64]]:
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
            return self._evaluate(self._model.simulateEIS, inputs)

        return self._evaluate(self._model.simulate, inputs)

    def _evaluate(
        self, func, inputs, calculate_grad=False
    ) -> dict[str, np.ndarray[np.float64]]:
        """
        Perform simulation using the specified method and handle exceptions.

        Parameters
        ----------
        func : callable
            The method to be used for simulation.
        inputs : Inputs
            Parameters for evaluation of the model.

        Returns
        -------
        dict[str, np.ndarray[np.float64]]
            The simulated model output.
        """
        try:
            if isinstance(self.model.solver, IDAKLUJax):
                sol = self._model.solver.get_vars(self.signal)(
                    self.domain_data, inputs
                )  # TODO: Add initial_state capabilities
            else:
                sol = func(
                    inputs,
                    self._domain_data,
                    initial_state=self.initial_state,
                )
        except (SolverError, ZeroDivisionError, RuntimeError, ValueError) as e:
            if isinstance(e, ValueError) and str(e) not in self.exception:
                raise  # Raise the error if it doesn't match the expected list
            error_out = {s: self.failure_output for s in self.signal}
            return (error_out, self.failure_output) if calculate_grad else error_out

        if self.eis:
            return sol

        if isinstance(self.model.solver, IDAKLUJax):
            return {signal: sol[:, i] for i, signal in enumerate(self.signal)}
        if calculate_grad:
            signals = self.signal + self.additional_variables
            return (
                {s: sol[s].data for s in signals},
                {s: sol[s].sensitivities for s in signals},
            )

        return {s: sol[s].data for s in (self.signal + self.additional_variables)}

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
        y, sens = self._evaluate(self._model.simulateS1, inputs, calculate_grad=True)

        if not any([np.isfinite(y[s]).any() for s in self.signal]):
            return y, sens

        # Extract the sensitivities for all signals at once
        param_keys = self.parameters.keys()
        dy = np.stack(
            [
                np.column_stack(
                    [sens[signal][key].toarray()[:, 0] for key in param_keys]
                )
                for signal in self.signal
            ],
            axis=1,
        )

        return y, dy
