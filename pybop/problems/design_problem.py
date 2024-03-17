import numpy as np
from pybop import BaseProblem


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
