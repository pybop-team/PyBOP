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

        # Set bounds
        self.bounds = dict(
            lower=[param.bounds[0] for param in self.parameters],
            upper=[param.bounds[1] for param in self.parameters],
        )

        # Sample from prior for x0
        if x0 is None:
            self.x0 = np.zeros(self.n_parameters)
            for i, param in enumerate(self.parameters):
                self.x0[i] = param.rvs(1)
        elif len(x0) != self.n_parameters:
            raise ValueError("x0 dimensions do not match number of parameters")

        # Add the initial values to the parameter definitions
        for i, param in enumerate(self.parameters):
            param.update(value=self.x0[i])

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
        Evaluate the model with the given parameters and return the signal and its derivatives.

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


class FittingProblem(BaseProblem):
    """
    Problem class for fitting (parameter estimation) problems.

    Extends `BaseProblem` with specifics for fitting a model to a dataset.

    Parameters
    ----------
    model : object
        The model to fit.
    parameters : list
        List of parameters for the problem.
    dataset : list
        List of data objects to fit the model to.
    signal : str, optional
        The signal to fit (default: "Voltage [V]").
    """

    def __init__(
        self,
        model,
        parameters,
        dataset,
        check_model=True,
        signal=["Voltage [V]"],
        init_soc=None,
        x0=None,
    ):
        super().__init__(parameters, model, check_model, signal, init_soc, x0)
        self._dataset = dataset.data

        # Check that the dataset contains time and current
        for name in ["Time [s]", "Current function [A]"] + self.signal:
            if name not in self._dataset:
                raise ValueError(f"expected {name} in list of dataset")

        self._time_data = self._dataset["Time [s]"]
        self.n_time_data = len(self._time_data)
        if np.any(self._time_data < 0):
            raise ValueError("Times can not be negative.")
        if np.any(self._time_data[:-1] >= self._time_data[1:]):
            raise ValueError("Times must be increasing.")

        target = [self._dataset[signal] for signal in self.signal]
        self._target = np.vstack(target).T
        if self.n_outputs == 1:
            if len(self._target) != self.n_time_data:
                raise ValueError("Time data and target data must be the same length.")
        else:
            if self._target.shape != (self.n_time_data, self.n_outputs):
                raise ValueError("Time data and target data must be the same shape.")

        # Add useful parameters to model
        if model is not None:
            self._model.signal = self.signal
            self._model.n_outputs = self.n_outputs
            self._model.n_time_data = self.n_time_data

        # Build the model
        if self._model._built_model is None:
            self._model.build(
                dataset=self._dataset,
                parameters=self.parameters,
                check_model=self.check_model,
                init_soc=self.init_soc,
            )

    def evaluate(self, x):
        """
        Evaluate the model with the given parameters and return the signal.

        Parameters
        ----------
        x : np.ndarray
            Parameter values to evaluate the model at.
        """

        y = np.asarray(self._model.simulate(inputs=x, t_eval=self._time_data))

        return y

    def evaluateS1(self, x):
        """
        Evaluate the model with the given parameters and return the signal and its derivatives.

        Parameters
        ----------
        x : np.ndarray
            Parameter values to evaluate the model at.
        """

        y, dy = self._model.simulateS1(
            inputs=x,
            t_eval=self._time_data,
        )

        return (np.asarray(y), np.asarray(dy))


class DesignProblem(BaseProblem):
    """
    Problem class for design optimization problems.

    Extends `BaseProblem` with specifics for applying a model to an experimental design.

    Parameters
    ----------
    model : object
        The model to apply the design to.
    parameters : list
        List of parameters for the problem.
    experiment : object
        The experimental setup to apply the model to.
    """

    def __init__(
        self,
        model,
        parameters,
        experiment,
        check_model=True,
        signal=["Voltage [V]"],
        init_soc=None,
        x0=None,
    ):
        super().__init__(parameters, model, check_model, signal, init_soc, x0)
        self.experiment = experiment

        # Build the model if required
        if experiment is not None:
            # Leave the build until later to apply the experiment
            self._model.parameters = self.parameters
            if self.parameters is not None:
                self._model.fit_keys = [param.name for param in self.parameters]

        elif self._model._built_model is None:
            self._model.build(
                experiment=self.experiment,
                parameters=self.parameters,
                check_model=self.check_model,
                init_soc=self.init_soc,
            )

    def evaluate(self, x):
        """
        Evaluate the model with the given parameters and return the signal.

        Parameters
        ----------
        x : np.ndarray
            Parameter values to evaluate the model at.
        """

        y = np.asarray(self._model.simulate(inputs=x, t_eval=self._time_data))

        return y

    def evaluateS1(self, x):
        """
        Evaluate the model with the given parameters and return the signal and its derivatives.

        Parameters
        ----------
        x : np.ndarray
            Parameter values to evaluate the model at.
        """

        y, dy = self._model.simulateS1(
            inputs=x,
            t_eval=self._time_data,
        )

        return (np.asarray(y), np.asarray(dy))
