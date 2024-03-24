import numpy as np
import pybop


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
        if not isinstance(dataset, pybop.Dataset):
            raise ValueError("Dataset must be a pybop Dataset object.")

        self._target = np.vstack([dataset[signal] for signal in self.signal]).T

    @property
    def model(self):
        return self._model


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
    dataset : Dataset
        Dataset object containing the data to fit the model to.
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
        self.x = self.x0

        # Check that the dataset contains time and current
        for name in ["Time [s]", "Current function [A]"] + self.signal:
            if name not in self._dataset:
                raise ValueError(f"Expected {name} in list of dataset")

        self._time_data = self._dataset["Time [s]"]
        self.n_time_data = len(self._time_data)
        if np.any(self._time_data < 0):
            raise ValueError("Times can not be negative.")
        if np.any(self._time_data[:-1] >= self._time_data[1:]):
            raise ValueError("Times must be increasing.")

        for signal in self.signal:
            if len(self._dataset[signal]) != self.n_time_data:
                raise ValueError(
                    f"Time data and {signal} data must be the same length."
                )
        self.set_target(dataset)

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

        Returns
        -------
        y : np.ndarray
            The model output y(t) simulated with inputs x.
        """
        if (x != self.x).any() and self._model.matched_parameters:
            for i, param in enumerate(self.parameters):
                param.update(value=x[i])

            self._model.rebuild(parameters=self.parameters)
            self.x = x

        y = np.asarray(self._model.simulate(inputs=x, t_eval=self._time_data))

        return y

    def evaluateS1(self, x):
        """
        Evaluate the model with the given parameters and return the signal and its derivatives.

        Parameters
        ----------
        x : np.ndarray
            Parameter values to evaluate the model at.

        Returns
        -------
        tuple
            A tuple containing the simulation result y(t) and the sensitivities dy/dx(t) evaluated
            with given inputs x.
        """
        if self._model.matched_parameters:
            raise RuntimeError(
                "Gradient not available when using geometric parameters."
            )

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

        # Add an example dataset for plotting comparison
        sol = self.evaluate(self.x0)
        self._time_data = sol[:, -1]
        self._target = sol[:, 0:-1]
        self._dataset = None

    def evaluate(self, x):
        """
        Evaluate the model with the given parameters and return the signal.

        Parameters
        ----------
        x : np.ndarray
            Parameter values to evaluate the model at.

        Returns
        -------
        y : np.ndarray
            The model output y(t) simulated with inputs x.
        """

        sol = self._model.predict(
            inputs=x,
            experiment=self.experiment,
            init_soc=self.init_soc,
        )

        if sol == [np.inf]:
            return sol

        else:
            predictions = [sol[signal].data for signal in self.signal + ["Time [s]"]]

            return np.vstack(predictions).T
