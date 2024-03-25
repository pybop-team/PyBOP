import pybop
import numpy as np


class BaseProblem:
    """
    Base class for defining a problem within the PyBOP framework, compatible with PINTS.

    Parameters
    ----------
    parameters : list
        List of parameters for the problem.
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
    x0 : np.ndarray, optional
        Initial parameter values (default: None).
    """

    def __init__(
        self,
        parameters,
        model=None,
        check_model=True,
        signal=["Voltage [V]"],
        additional_variables=[],
        init_soc=None,
        x0=None,
    ):
        self.parameters = parameters
        self._model = model
        self.check_model = check_model
        if isinstance(signal, str):
            signal = [signal]
        elif not all(isinstance(item, str) for item in signal):
            raise ValueError("Signal should be either a string or list of strings.")
        self.signal = signal
        self.init_soc = init_soc
        self.x0 = x0
        self.n_parameters = len(self.parameters)
        self.n_outputs = len(self.signal)
        self._time_data = None
        self._target = None

        if isinstance(model, pybop.BaseModel):
            self.additional_variables = additional_variables
        else:
            self.additional_variables = []

        # Set bounds (for all or no parameters)
        all_unbounded = True  # assumption
        self.bounds = {"lower": [], "upper": []}
        for param in self.parameters:
            if param.bounds is not None:
                self.bounds["lower"].append(param.bounds[0])
                self.bounds["upper"].append(param.bounds[1])
                all_unbounded = False
            else:
                self.bounds["lower"].append(-np.inf)
                self.bounds["upper"].append(np.inf)
        if all_unbounded:
            self.bounds = None

        # Set initial standard deviation (for all or no parameters)
        all_have_sigma = True  # assumption
        self.sigma0 = []
        for param in self.parameters:
            if hasattr(param.prior, "sigma"):
                self.sigma0.append(param.prior.sigma)
            else:
                all_have_sigma = False
        if not all_have_sigma:
            self.sigma0 = None

        # Sample from prior for x0
        if x0 is None:
            self.x0 = np.zeros(self.n_parameters)
            for i, param in enumerate(self.parameters):
                self.x0[i] = param.rvs(1)
        elif len(x0) != self.n_parameters:
            raise ValueError("x0 dimensions do not match number of parameters")

        # Add the initial values to the parameter definitions
        for i, param in enumerate(self.parameters):
            param.update(initial_value=self.x0[i])

    def evaluate(self, x):
        """
        Evaluate the model with the given parameters and return the signal.

        Parameters
        ----------
        x : np.ndarray
            Parameter values to evaluate the model at.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def evaluateS1(self, x):
        """
        Evaluate the model with the given parameters and return the signal and
        its derivatives.

        Parameters
        ----------
        x : np.ndarray
            Parameter values to evaluate the model at.

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

    def target(self):
        """
        Return the target dataset.

        Returns
        -------
        np.ndarray
            The target dataset array.
        """
        return self._target

    @property
    def model(self):
        return self._model
