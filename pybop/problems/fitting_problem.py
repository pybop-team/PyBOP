import numpy as np

import pybop

"""
TODO:
1. Allow multiple datasets, one model
    Include weights function: func[list[errors]]->scalar error
2. Allow multiple models
3. Allow serial, parallel dispatch
"""


class FittingProblem(pybop.BaseProblem):
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
        The variable used for fitting (default: "Voltage [V]").
    additional_variables : List[str], optional
        Additional variables to observe and store in the solution (default additions are: ["Time [s]"]).
    init_soc : float, optional
        Initial state of charge (default: None).
    x0 : np.ndarray, optional
        Initial parameter values (default: None).
    """

    def __init__(
        self,
        model,
        parameters,
        dataset,
        check_model=True,
        signal=["Voltage [V]"],
        additional_variables=[],
        init_soc=None,
        x0=None,
    ):
        # Add time and remove duplicates
        additional_variables.extend(["Time [s]"])
        additional_variables = list(set(additional_variables))

        if not hasattr(model, "__len__"):
            model = [model] * dataset.n_datasets
        # TODO check valid (1,1), (1,n), (n,1), (n,n) model/dataset pairings

        super().__init__(
            parameters, model, check_model, signal, additional_variables, init_soc, x0
        )
        self._dataset = dataset
        self.x = self.x0

        # Check that the dataset contains time and current
        dataset.check(self.signal + ["Current function [A]"])

        # Unpack time and target data
        self.set_target(dataset)
        try:
            self._time_data = [data["Time [s]"] for data in self._dataset]
        except TypeError:
            self._time_data = [self._dataset["Time [s]"]]

        # Add useful parameters to model
        if model is not None:
            for thismodel, thisdataset in zip(self._model, self._dataset):
                thismodel.signal = self.signal
                thismodel.additional_variables = self.additional_variables
                # TODO generalise to allow different numbers of parameters, outputs
                thismodel.n_parameters = self.n_parameters
                thismodel.n_outputs = self.n_outputs

                # Build the model from scratch
                if thismodel._built_model is not None:
                    thismodel._model_with_set_params = None
                    thismodel._built_model = None
                    thismodel._built_initial_soc = None
                    thismodel._mesh = None
                    thismodel._disc = None
                # TODO generalise to allow different initial SOCs
                thismodel.build(
                    dataset=thisdataset,
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
        y : list[dict[str, np.ndarray[np.float64]]
            List of model outputs y(t) simulated with inputs x, with
            one dict for each dataset
        """
        y = []
        for model, ts in zip(self._model, self._time_data):
            if np.any(x != self.x) and model.matched_parameters:
                for i, param in enumerate(self.parameters):
                    param.update(value=x[i])

                model.rebuild(parameters=self.parameters)
                self.x = x

            y.append(model.simulate(inputs=x, t_eval=ts))

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
            with given inputs x, for each dataset.
        """
        if self._model.matched_parameters:
            raise RuntimeError(
                "Gradient not available when using geometric parameters."
            )

        sims = [
            model.simulateS1(inputs=x, t_eval=ts)
            for ts, model in zip(self._time_data, self._model)
        ]
        y = [s[0] for s in sims]
        dy = [s[1] for s in sims]

        return (y, np.asarray(dy))

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
        if not isinstance(dataset, (pybop.Dataset, pybop.DatasetCollection)):
            raise ValueError("Dataset must be a pybop Dataset object.")

        # List, with one signal/target dict per dataset
        try:
            self._target = [
                {signal: dataset[signal][i] for signal in self.signal}
                for i in range(dataset.n_datasets)
            ]
        except AttributeError:
            self._target = [{signal: dataset[signal] for signal in self.signal}]
