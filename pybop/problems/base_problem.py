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
    signal: List[str]
      The signal to observe.
    additional_variables : List[str], optional
        Additional variables to observe and store in the solution (default: []).
    init_soc : float, optional
        Initial state of charge (default: None).
    """

    def __init__(
        self,
        parameters,
        model=None,
        check_model=True,
        signal=None,
        additional_variables=None,
        init_soc=None,
    ):
        # Check if parameters is a list of pybop.Parameter objects
        if additional_variables is None:
            additional_variables = []
        if signal is None:
            signal = ["Voltage [V]"]
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
        self._model = model
        self.check_model = check_model
        if isinstance(signal, str):
            signal = [signal]
        elif not all(isinstance(item, str) for item in signal):
            raise ValueError("Signal should be either a string or list of strings.")
        self.signal = signal
        self.init_soc = init_soc
        self.n_outputs = len(self.signal)
        self._time_data = None
        self._target = None

        if isinstance(model, BaseModel):
            self.additional_variables = additional_variables
        else:
            self.additional_variables = []

    @property
    def n_parameters(self):
        return len(self.parameters)

    def evaluate(self, inputs: Inputs):
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

    def time_data(self):
        """
        Returns the time data.

        Returns
        -------
        np.ndarray
            The time array.
        """
        return self._time_data

    def get_target(self):
        """
        Return the target dataset.

        Returns
        -------
        np.ndarray
            The target dataset array.
        """
        return self._target

    def set_target(self, dataset):
        """
        Set the target dataset.

        Parameters
        ----------
        target : np.ndarray
            The target dataset array.
        """
        if self.signal is None:
            raise ValueError("Signal must be defined to set target.")
        if not isinstance(dataset, Dataset):
            raise ValueError("Dataset must be a pybop Dataset object.")

        self._target = {signal: dataset[signal] for signal in self.signal}

    @property
    def model(self):
        return self._model
