import numpy as np


class BaseProblem:
    """
    Defines the PyBOP base problem, following the PINTS interface.
    """

    def __init__(
        self,
        parameters,
        model=None,
        check_model=True,
        init_soc=None,
        x0=None,
    ):
        self._model = model
        self.check_model = check_model
        self.parameters = parameters
        self.init_soc = init_soc
        self.x0 = x0
        self.n_parameters = len(self.parameters)

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
        self.parameters = {o.name: o.value for o in parameters}

    def evaluate(self, parameters):
        """
        Evaluate the model with the given parameters and return the signal.
        """
        raise NotImplementedError

    def evaluateS1(self, parameters):
        """
        Evaluate the model with the given parameters and return the signal and
        its derivatives.
        """
        raise NotImplementedError


class FittingProblem(BaseProblem):
    """
    Defines the problem class for a fitting (parameter estimation) problem.
    """

    def __init__(
        self,
        model,
        parameters,
        dataset,
        signal="Terminal voltage [V]",
        check_model=True,
        init_soc=None,
        x0=None,
    ):
        super().__init__(parameters, model, check_model, init_soc, x0)
        if model is not None:
            self._model.signal = signal
        self.signal = signal
        self._dataset = {o.name: o for o in dataset}
        self.n_outputs = len([self.signal])

        # Check that the dataset contains time and current
        for name in ["Time [s]", "Current function [A]", signal]:
            if name not in self._dataset:
                raise ValueError(f"expected {name} in list of dataset")

        self._time_data = self._dataset["Time [s]"].data
        self.n_time_data = len(self._time_data)
        self._target = self._dataset[signal].data

        if np.any(self._time_data < 0):
            raise ValueError("Times can not be negative.")
        if np.any(self._time_data[:-1] >= self._time_data[1:]):
            raise ValueError("Times must be increasing.")

        if len(self._target) != len(self._time_data):
            raise ValueError("Time data and signal data must be the same length.")

        # Build the model
        if self._model._built_model is None:
            self._model.build(
                dataset=self._dataset,
                parameters=self.parameters,
                check_model=self.check_model,
                init_soc=self.init_soc,
            )

    def evaluate(self, parameters):
        """
        Evaluate the model with the given parameters and return the signal.
        """

        y = np.asarray(self._model.simulate(inputs=parameters, t_eval=self._time_data))

        return y

    def evaluateS1(self, parameters):
        """
        Evaluate the model with the given parameters and return the signal and
        its derivatives.
        """
        for i, key in enumerate(self.parameters):
            self.parameters[key] = parameters[i]

        y, dy = self._model.simulateS1(
            inputs=self.parameters,
            t_eval=self._time_data,
        )

        return (np.asarray(y), np.asarray(dy))


class DesignProblem(BaseProblem):
    """
    Defines the problem class for a design optimiation problem.
    """

    def __init__(
        self,
        model,
        parameters,
        experiment,
        check_model=True,
        init_soc=None,
        x0=None,
    ):
        super().__init__(parameters, model, check_model, init_soc, x0)
        self.experiment = experiment
        self._target = None

        # Build the model if required
        if experiment is not None:
            # Leave the build until later to apply the experiment
            self._model.parameters = self.parameters

        elif self._model._built_model is None:
            self._model.build(
                experiment=self.experiment,
                parameters=self.parameters,
                check_model=self.check_model,
                init_soc=self.init_soc,
            )

    def evaluate(self, parameters):
        """
        Evaluate the model with the given parameters and return the signal.
        """

        y = np.asarray(self._model.simulate(inputs=parameters, t_eval=self._time_data))

        return y

    def evaluateS1(self, parameters):
        """
        Evaluate the model with the given parameters and return the signal and
        its derivatives.
        """
        for i, key in enumerate(self.parameters):
            self.parameters[key] = parameters[i]

        y, dy = self._model.simulateS1(
            inputs=self.parameters,
            t_eval=self._time_data,
        )

        return (np.asarray(y), np.asarray(dy))
