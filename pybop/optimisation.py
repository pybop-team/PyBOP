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
            cost_function=self.cost,
            x0=self.x0,
            bounds=self.bounds,
        )

        return results
