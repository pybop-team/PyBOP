class Optimisation:
    """
    Optimisation class for PyBOP.
    """

    def __init__(
        self,
        cost,
        optimiser,
        verbose=False,
    ):
        self.cost = cost
        self.optimiser = optimiser
        self.verbose = verbose
        self.x0 = cost._problem.x0
        self.bounds = cost._problem.bounds
        self.fit_parameters = {}

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

        # Update the parameter dictionary
        for i, key in enumerate(self.cost._problem.fit_parameters):
            self.fit_parameters[key] = x[i]

        # Compute cost
        res = self.cost.compute(self.fit_parameters)

        if self.verbose:
            print("Parameter estimates: ", self.cost._problem.parameters, "\n")

        return res
