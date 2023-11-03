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
        parameters,
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
        self.parameters = parameters
        self.x0 = x0
        self.dataset = {o.name: o for o in dataset}
        self.signal = signal
        self.n_parameters = len(self.parameters)
        self.verbose = verbose

        # Check that the dataset contains time and current
        for name in ["Time [s]", "Current function [A]"]:
            if name not in self.dataset:
                raise ValueError(f"expected {name} in list of dataset")

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

        # Add the initial values to the parameter definitions
        for i, param in enumerate(self.parameters):
            param.update(value=self.x0[i])

        self.fit_parameters = {o.name: o for o in parameters}
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
            cost_function=self.cost_function,
            x0=self.x0,
            bounds=self.bounds,
        )

        return results

    def cost_function(self, x, grad=None):
        """
        Compute a model prediction and associated value of the cost.
        """

        # Unpack the target dataset
        target = self.dataset[self.signal].data

        # Update the parameter dictionary
        for i, key in enumerate(self.fit_parameters):
            self.fit_parameters[key] = x[i]

        # Make prediction
        prediction = self.model.simulate(
            inputs=self.fit_parameters, t_eval=self.model.time_data
        )[self.signal].data

        # Compute cost
        res = self.cost.compute(prediction, target)

        if self.verbose:
            print("Parameter estimates: ", self.parameters.value, "\n")

        return res
