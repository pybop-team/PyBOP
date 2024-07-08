import pybop


class StandaloneCost(pybop.BaseCost):
    """
    A standalone cost function example that inherits from pybop.BaseCost.

    This class represents a simple cost function without a problem object, used for demonstration purposes.
    It is a quadratic function of one variable with a constant term, defined by
    the formula: cost(x) = x^2 + 42.

    Parameters
    ----------
    problem : object, optional
        A dummy problem instance used to initialize the superclass. This is not
        used in the current class but is accepted for compatibility with the
        BaseCost interface.
    parameters : pybop.Parameters
        A pybop.Parameters object storing a dictionary of parameters and their
        properties, for example their initial value and bounds.

    Methods
    -------
    __call__(x, grad=None)
        Calculate the cost for a given parameter value.
    """

    def __init__(self, problem=None):
        """
        Initialise the StandaloneCost class with optional problem instance.

        The problem object is not utilised in this subclass. The parameters, including
        their initial value and bounds, are defined within this standalone cost object.
        """
        super().__init__(problem)

        self.parameters = pybop.Parameters(
            pybop.Parameter(
                "x",
                initial_value=4.2,
                bounds=[-1, 10],
            ),
        )
        self.x0 = self.parameters.initial_value()

    def _evaluate(self, inputs, grad=None):
        """
        Calculate the cost for a given parameter value.

        The cost function is defined as cost(x) = x^2 + 42, where x is the
        parameter value.

        Parameters
        ----------
        inputs : Dict
            The parameters for which to evaluate the cost.
        grad : array-like, optional
            Unused parameter, present for compatibility with gradient-based
            optimizers.

        Returns
        -------
        float
            The calculated cost value for the given parameter.
        """

        return inputs["x"] ** 2 + 42
