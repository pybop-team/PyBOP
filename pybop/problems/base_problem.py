from typing import Optional

import numpy as np
from pybamm import IDAKLUSolver

from pybop import BaseModel, Dataset, Parameter, Parameters
from pybop.parameters.parameter import Inputs


class BaseProblem:
    """
    Base class for defining a problem within the PyBOP framework, compatible with PINTS.

    Parameters
    ----------
    parameters : pybop.Parameter or pybop.Parameters
        An object or list of the parameters for the problem.
    model : object, optional
        The model to be used for the problem (default: None).
    check_model : bool, optional
        Flag to indicate if the model should be checked (default: True).
    signal: list[str]
      The signal to observe.
    additional_variables : list[str], optional
        Additional variables to observe and store in the solution (default: []).
    initial_state : dict, optional
        A valid initial state (default: None).
    """

    def __init__(
        self,
        parameters: Parameters,
        model: Optional[BaseModel] = None,
        check_model: bool = True,
        signal: Optional[list[str]] = None,
        additional_variables: Optional[list[str]] = None,
        initial_state: Optional[dict] = None,
    ):
        signal = signal or ["Voltage [V]"]
        if isinstance(signal, str):
            signal = [signal]
        elif not all(isinstance(item, str) for item in signal):
            raise ValueError("Signal should be either a string or list of strings.")

        # Check if parameters is a list of pybop.Parameter objects
        if isinstance(parameters, list):
            if all(isinstance(param, Parameter) for param in parameters):
                parameters = Parameters(*parameters)
            else:
                raise TypeError(
                    "All elements in the list must be pybop.Parameter objects."
                )
        # Check if parameters is a single pybop.Parameter object
        elif isinstance(parameters, Parameter):
            parameters = Parameters(parameters)
        # Check if parameters is already a pybop.Parameters object
        elif not isinstance(parameters, Parameters):
            raise TypeError(
                "The input parameters must be a pybop Parameter, a list of pybop.Parameter objects, or a pybop Parameters object."
            )

        self.parameters = parameters
        self.parameters.reset_initial_value()

        self._model = model
        self.eis = False
        self.domain = "Time [s]"
        self.check_model = check_model
        self.signal = signal or ["Voltage [V]"]
        self.set_initial_state(initial_state)
        self._dataset = None
        self._domain_data = None
        self._target = None
        self.verbose = False
        self.failure_output = np.asarray([np.inf])
        if isinstance(self._model, BaseModel):
            self.eis = self.model.eis
            self.domain = "Frequency [Hz]" if self.eis else "Time [s]"

        # Add domain and remove duplicates
        self.additional_variables = additional_variables or []
        self.additional_variables.extend([self.domain, "Current [A]"])
        self.additional_variables = list(set(self.additional_variables))

        # If model.solver is IDAKLU, set output vars for improved performance
        self.output_vars = tuple(self.signal + self.additional_variables)
        if self._model is not None and isinstance(self._model.solver, IDAKLUSolver):
            self._solver_copy = self._model.solver.copy()
            self._model.solver = IDAKLUSolver(
                atol=self._solver_copy.atol,
                rtol=self._solver_copy.rtol,
                root_method=self._solver_copy.root_method,
                root_tol=self._solver_copy.root_tol,
                extrap_tol=self._solver_copy.extrap_tol,
                options=self._solver_copy._options,  # noqa: SLF001
                output_variables=self.output_vars,
            )

    def set_initial_state(self, initial_state: Optional[dict] = None):
        """
        Set the initial state to be applied to evaluations of the problem.

        Parameters
        ----------
        initial_state : dict, optional
            A valid initial state (default: None).
        """
        self.initial_state = initial_state

    @property
    def n_parameters(self):
        return len(self.parameters)

    @property
    def n_outputs(self):
        return len(self.signal)

    def evaluate(self, inputs: Inputs, eis=False):
        """
        Evaluate the model with the given parameters and return the signal.

        Parameters
        ----------
        inputs : Inputs
            Parameters for evaluation of the model.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def evaluateS1(self, inputs: Inputs):
        """
        Evaluate the model with the given parameters and return the signal and
        its derivatives.

        Parameters
        ----------
        inputs : Inputs
            Parameters for evaluation of the model.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def get_target(self):
        """
        Return the target dataset.

        Returns
        -------
        np.ndarray
            The target dataset array.
        """
        return self._target

    def set_target(self, dataset: Dataset):
        """
        Set the target dataset.

        Parameters
        ----------
        target : Dataset
            The target dataset array.
        """
        if self.signal is None:
            raise ValueError("Signal must be defined to set target.")
        if not isinstance(dataset, Dataset):
            raise ValueError("Dataset must be a pybop Dataset object.")

        self._domain_data = dataset[self.domain]
        self._target = {signal: dataset[signal] for signal in self.signal}

    @property
    def model(self):
        return self._model

    @property
    def target(self):
        return self._target

    @property
    def domain_data(self):
        return self._domain_data

    @domain_data.setter
    def domain_data(self, domain_data):
        self._domain_data = domain_data

    @property
    def dataset(self):
        return self._dataset
