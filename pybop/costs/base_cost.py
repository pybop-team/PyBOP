from typing import Optional, Union

from pybop import BaseProblem
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
    _has_separable_problem : bool
        If True, the problem is separable from the cost function and will be
        evaluated in advance of the call to self.compute() (default: False).
    """

    def __init__(self, problem: Optional[BaseProblem] = None):
        self.parameters = Parameters()
        self.transformation = None
        self.problem = problem
        self.verbose = False
        self._has_separable_problem = False
        self.update_capacity = False
        self.y = None
        self.dy = None
        self.set_fail_gradient()
        if isinstance(self.problem, BaseProblem):
            self._target = self.problem.target
            self.parameters.join(self.problem.parameters)
            self.n_outputs = self.problem.n_outputs
            self.signal = self.problem.signal
            self.transformation = self.parameters.construct_transformation()
            self._has_separable_problem = True

    @property
    def n_parameters(self):
        return len(self.parameters)

    @property
    def has_separable_problem(self):
        return self._has_separable_problem

    @property
    def target(self):
        return self._target

    def __call__(self, inputs: Union[Inputs, list], calculate_grad=False):
        """
        This method calls the forward model via problem.evaluate(inputs),
        and computes the cost for the given output by calling self.compute(inputs).

        Parameters
        ----------
        inputs : Inputs or array-like
            The parameters for which to compute the cost and gradient.

        Returns
        -------
        float
            The calculated cost function value.

        Raises
        ------
        ValueError
            If an error occurs during the calculation of the cost.
        """
        if self.transformation:
            p = self.transformation.to_model(inputs)
        inputs = self.parameters.verify(p if self.transformation else inputs)

        try:
            if self._has_separable_problem:
                if calculate_grad is True:
                    self.y, self.dy = self.problem.evaluateS1(
                        inputs,
                        # calculate_grad=calculate_grad,
                    )
                else:
                    self.y = self.problem.evaluate(
                        inputs,
                        update_capacity=self.update_capacity,
                    )
            return self.compute(inputs, calculate_grad=calculate_grad)

        except NotImplementedError as e:
            raise e

        except Exception as e:
            raise ValueError(f"Error in cost calculation: {e}") from e

    def compute(self, inputs: Inputs, calculate_grad=False):
        """
        Compute the cost and  if `calculate_grad` is True, its gradient with
        respect to the parameters.

        This method only computes the cost, without calling the `problem.evaluate()`.
        This method must be implemented by subclasses.

        Parameters
        ----------
        inputs : Inputs
            The parameters for which to compute the cost.

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

    def set_fail_gradient(self, de: float = 1.0):
        """
        Set the fail gradient to a specified value.

        The fail gradient is used if an error occurs during the calculation
        of the gradient. This method allows updating the default gradient value.

        Parameters
        ----------
        de : float
            The new fail gradient value to be used.
        """
        if not isinstance(de, float):
            de = float(de)
        self._de = de

    def verify_prediction(self, y):
        """
        Verify that the prediction matches the target data.

        Parameters
        ----------
        y : dict
            The model predictions.

        Returns
        -------
        bool
            True if the prediction matches the target data, otherwise False.
        """
        if any(
            len(y.get(key, [])) != len(self._target.get(key, [])) for key in self.signal
        ):
            return False

        return True
