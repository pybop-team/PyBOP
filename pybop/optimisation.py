import pybop
import numpy as np


class Optimisation:
    """
    Optimisation class for PyBOP.
    """

    def __init__(
        self,
        cost,
        model,
        optimiser,
        fit_parameters,
        x0=None,
        dataset=None,
        signal=None,
        check_model=True,
        init_soc=None,
        verbose=False,
    ):
        self.cost = cost
        self.model = model
        self.optimiser = optimiser
        self.x0 = x0
        self.dataset = {o.name: o for o in dataset}
        self.signal = signal
        self.verbose = verbose
        self.fit_dict = {}
        self.fit_parameters = {o.name: o for o in fit_parameters}
        self.model.n_parameters = len(self.fit_dict)

        # Check that the dataset contains time and current
        for name in ["Time [s]", "Current function [A]"]:
            if name not in self.dataset:
                raise ValueError(f"expected {name} in list of dataset")

        # Set bounds
        self.bounds = dict(
            lower=[self.fit_parameters[param].bounds[0] for param in self.fit_parameters],
            upper=[self.fit_parameters[param].bounds[1] for param in self.fit_parameters],
        )

        # Sample from prior for x0
        if x0 is None:
            self.x0 = np.zeros(len(self.fit_parameters))
            for i, j in enumerate(self.fit_parameters):
                self.x0[i] = self.fit_parameters[j].prior.rvs(1)[
                    0
                ]  # Updt to capture dimensions per parameter

        # Add the initial values to the parameter definitions
        for i, param in enumerate(self.fit_parameters):
            self.fit_dict[param] = {param: self.x0[i]}

        # Build model with dataset and fitting parameters
        self.model.build(
            dataset=self.dataset,
            fit_parameters=self.fit_parameters,
            check_model=check_model,
            init_soc=init_soc,
        )

    def run(self):
        """
        Run the optimisation algorithm.
        """

        results = self.optimiser.optimise(
            self.cost_function,  # lambda x, grad: self.cost_function(x, grad),
            self.x0,
            self.bounds,
        )

        return results

    def cost_function(self, x, grad=None):
        """
        Compute a model prediction and associated value of the cost.
        """

        # Unpack the target dataset
        target = self.dataset[self.signal].data

        # Update the parameter dictionary
        for i, param in enumerate(self.fit_dict):
            self.fit_dict[param] = x[i]

        # Make prediction
        prediction = self.model.predict(inputs=self.fit_dict)[self.signal].data

        # Add simulation error handling here

        # Compute cost
        res = self.cost.compute(prediction, target)

        if self.verbose:
            print("Parameter estimates: ", self.parameters.value, "\n")

        return res
