from pybop import BaseProblem, ComposedTransformation, IdentityTransformation
from pybop.parameters.parameter import Inputs, Parameters


class BaseCost:
    """
    Base class for defining cost functions.

    This class is intended to be subclassed to create specific cost functions
    for evaluating model predictions against a set of data. The cost function
    quantifies the goodness-of-fit between the model predictions and the
    observed data, with a lower cost value indicating a better fit.

    Parameters
    ----------
    problem : object
        A problem instance containing the data and functions necessary for
        evaluating the cost function.
    _target : array-like
        An array containing the target data to fit.
    n_outputs : int
        The number of outputs in the model.
    """

    def __init__(self, problem=None):
        self.parameters = Parameters()
        self.transformation = None
        self.problem = problem
        if isinstance(self.problem, BaseProblem):
            self._target = self.problem._target
            self.parameters.join(self.problem.parameters)
            self.n_outputs = self.problem.n_outputs
            self.signal = self.problem.signal
            self.transformation = self.construct_transformation()

    def construct_transformation(self):
        """
        Create a ComposedTransformation object from the individual parameters transformations.
        """
        transformations = self.parameters.get_transformations()
        if not transformations or all(t is None for t in transformations):
            return None

        valid_transformations = [
            t if t is not None else IdentityTransformation() for t in transformations
        ]
        return ComposedTransformation(valid_transformations)

    def __call__(self, x):
        """
        Call the evaluate function for a given set of parameters.
        """
        if self.transformation:
            p = self.transformation.to_model(x)
            return self.evaluate(p)
        else:
            return self.evaluate(x)

    def evaluate(self, x):
        """
        Call the evaluate function for a given set of parameters.

        Parameters
        ----------
        x : array-like
            The parameters for which to evaluate the cost.

        Returns
        -------
        float
            The calculated cost function value.

        Raises
        ------
        ValueError
            If an error occurs during the calculation of the cost.
        """
        inputs = self.parameters.verify(x)

        try:
            return self._evaluate(inputs)

        except NotImplementedError as e:
            raise e

        except Exception as e:
            raise ValueError(f"Error in cost calculation: {e}") from e

    def _evaluate(self, inputs: Inputs):
        """
        Calculate the cost function value for a given set of parameters.

        This method must be implemented by subclasses.

        Parameters
        ----------
        inputs : Inputs
            The parameters for which to evaluate the cost.

        Returns
        -------
        float
            The calculated cost function value.

        Raises
        ------
        NotImplementedError
            If the method has not been implemented by the subclass.
        """
        raise NotImplementedError

    def evaluateS1(self, x):
        """
        Call _evaluateS1 for a given set of parameters.

        Parameters
        ----------
        x : array-like
            The parameters for which to compute the cost and gradient.

        Returns
        -------
        tuple
            A tuple containing the cost and the gradient. The cost is a float,
            and the gradient is an array-like of the same length as `x`.

        Raises
        ------
        ValueError
            If an error occurs during the calculation of the cost or gradient.
        """
        inputs = self.parameters.verify(x)

        try:
            if self.transformation:
                p = self.transformation.to_model(inputs)
                return self._evaluateS1(p)
            else:
                return self._evaluateS1(inputs)

        except NotImplementedError as e:
            raise e

        except Exception as e:
            raise ValueError(f"Error in cost calculation: {e}") from e

    def _evaluateS1(self, inputs: Inputs):
        """
        Compute the cost and its gradient with respect to the parameters.

        Parameters
        ----------
        inputs : Inputs
            The parameters for which to compute the cost and gradient.

        Returns
        -------
        tuple
            A tuple containing the cost and the gradient. The cost is a float,
            and the gradient is an array-like of the same length as `x`.

        Raises
        ------
        NotImplementedError
            If the method has not been implemented by the subclass.
        """
        raise NotImplementedError

    @property
    def n_parameters(self):
        return len(self.parameters)
