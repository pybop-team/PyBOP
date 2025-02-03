from typing import Optional, Union

import numpy as np
from SALib.analyze import sobol
from SALib.sample.sobol import sample

from pybop import BaseProblem
from pybop._utils import add_spaces
from pybop.parameters.parameter import Inputs, Parameters


class BaseCost:
    """
    Base class for defining cost functions.

    Parameters
    ----------
    problem : object
        A problem instance containing the data and functions necessary for
        evaluating the cost function.
    target : array-like
        An array containing the target data to fit.
    n_outputs : int
        The number of outputs in the model.
    has_separable_problem : bool
        If True, the problem is separable from the cost function and will be
        evaluated in advance of the call to self.compute() (default: False).
    _de : float
        The gradient of the cost function to use if an error occurs during
        evaluation. Defaults to 1.0.
    minimising : bool, optional, default=True
        If False, switches the sign of the cost and gradient to perform maximisation
        instead of minimisation.
    """

    class DeferredPrediction:
        """
        Class used to indicate a prediction has yet to be, but is expected during
        a cost call.
        """

        pass

    def __init__(self, problem: Optional[BaseProblem] = None):
        self._parameters = Parameters()
        self._transformation = None
        self.problem = problem
        self.verbose = False
        self._has_separable_problem = False
        self.y = None
        self.dy = None
        self._de = 1.0
        self.minimising = True
        if isinstance(self.problem, BaseProblem):
            self._target = self.problem.target
            self._parameters.join(self.problem.parameters)
            self.n_outputs = self.problem.n_outputs
            self.signal = self.problem.signal
            self._transformation = self._parameters.construct_transformation()
            self._has_separable_problem = True
            self.grad_fail = None
            self.set_fail_gradient()

    def __call__(
        self,
        inputs: Union[Inputs, list, np.ndarray],
        calculate_grad: bool = False,
        apply_transform: bool = False,
        for_optimiser: bool = False,
    ) -> Union[float, list, tuple[float, np.ndarray], list[tuple[float, np.ndarray]]]:
        """
        Compute cost and optional gradient for given input parameters.

        Parameters
        ----------
        inputs : Union[Inputs, list, np.ndarray]
            Input parameters for cost computation. Supports list-like evaluation of
            multiple input values, shaped [N,M] where N is the number of input positions
            to evaluate and M is the number of inputs for the underlying model (i.e. parameters).
        calculate_grad : bool, default=False
            If True, both the cost and gradient will be computed. Otherwise, only the
            cost is computed.
        apply_transform : bool, default=False
            If True, applies a transformation to the inputs before evaluating the model.
        for_optimiser : bool, default=False
            If True, adjusts output sign based on minimisation/maximisation setting.

        Returns
        -------
        Union[float, list, tuple[float, np.ndarray], list[tuple[float, np.ndarray]]]
            - Single input, no gradient: float
            - Multiple inputs, no gradient: list[float]
            - Single input with gradient: tuple[float, np.ndarray]
            - Multiple inputs with gradient: list[tuple[float, np.ndarray]]
        """
        # Convert dict to list for sequential computations
        if isinstance(inputs, dict):
            inputs = list(inputs.values())
        input_list = np.atleast_2d(inputs)

        minimising = self.minimising or not for_optimiser
        sign = 1 if minimising else -1

        results = []
        for input_val in input_list:
            result = self._evaluate_single_input(
                input_val,
                calculate_grad=calculate_grad,
                apply_transform=apply_transform,
                sign=sign,
            )
            results.append(result)

        return results[0] if len(input_list) == 1 else results

    def _evaluate_single_input(
        self,
        input_value: Union[Inputs, np.ndarray],
        calculate_grad: bool,
        apply_transform: bool,
        sign: int,
    ) -> Union[float, tuple[float, np.ndarray]]:
        """Evaluate cost (and optional gradient) for a single input."""
        # Setup input transformation
        self.has_transform = self.transformation is not None and apply_transform
        model_inputs = self.parameters.verify(self._apply_transformations(input_value))
        self.parameters.update(values=list(model_inputs.values()))

        if self._has_separable_problem:
            return self._evaluate_separable_problem(
                input_value, calculate_grad=calculate_grad, sign=sign
            )

        return self._evaluate_non_separable_problem(
            calculate_grad=calculate_grad, sign=sign
        )

    def _evaluate_separable_problem(
        self, input_value: Union[Inputs, np.ndarray], calculate_grad: bool, sign: int
    ) -> Union[float, tuple[float, np.ndarray]]:
        """Evaluation for separable problems."""
        if calculate_grad:
            y, dy = self.problem.evaluateS1(self.problem.parameters.as_dict())
            cost, grad = self.compute(y, dy=dy)

            if self.has_transform and np.isfinite(cost):
                jac = self.transformation.jacobian(input_value)
                grad = np.matmul(grad, jac)

            return cost * sign, grad * sign

        y = self.problem.evaluate(self.problem.parameters.as_dict())
        return self.compute(y, dy=None) * sign

    def _evaluate_non_separable_problem(
        self, calculate_grad: bool, sign: int
    ) -> Union[float, tuple[float, np.ndarray]]:
        """Evaluation for non-separable problems."""
        y = self.DeferredPrediction
        dy = self.DeferredPrediction if calculate_grad else None

        if calculate_grad:
            cost, grad = self.compute(y, dy=dy)
            return cost * sign, grad * sign

        return self.compute(y, dy=dy) * sign

    def _apply_transformations(self, inputs):
        """Apply transformation if needed"""
        return self.transformation.to_model(inputs) if self.has_transform else inputs

    def compute(self, y: dict, dy: Optional[np.ndarray]):
        """
        Compute the cost and, if dy is not None, its gradient with respect to the
        parameters.

        This method only computes the cost, without calling the `problem.evaluate()`.
        This method must be implemented by subclasses.

        Parameters
        ----------
        y : dict
            The dictionary of predictions with keys designating the signals for fitting.
        dy : np.ndarray, optional
            The corresponding gradient with respect to the parameters for each signal.

        Raises
        ------
        NotImplementedError
            If the method has not been implemented by the subclass.
        """
        raise NotImplementedError

    def sensitivity_analysis(self, n_samples: int = 256):
        """
        Computes the parameter sensitivities on the cost function using
        SOBOL analyse from the SALib module [1].

        Parameters
        ----------
        n_samples : int, optional
            Number of samples for SOBOL sensitivity analysis,
            performs best as order of 2, i.e. 128, 256, etc.

        References
        ----------
        .. [1] Iwanaga, T., Usher, W., & Herman, J. (2022). Toward SALib 2.0:
               Advancing the accessibility and interpretability of global sensitivity
               analyses. Socio-Environmental Systems Modelling, 4, 18155. doi:10.18174/sesmo.18155

        Returns
        -------
        Sensitivities : dict
        """

        salib_dict = {
            "names": self.parameters.keys(),
            "bounds": self.parameters.get_bounds_for_plotly(),
            "num_vars": len(self.parameters.keys()),
        }

        param_values = sample(salib_dict, n_samples)
        return sobol.analyze(salib_dict, np.asarray(self.__call__(param_values)))

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
        self.grad_fail = self._de * np.ones(self.n_parameters)

    def verify_prediction(self, y: dict):
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

    def join_parameters(self, parameters):
        """
        Setter for joining parameters. This method sets the fail gradient if the join adds parameters.
        """
        original_n_params = self.n_parameters
        self._parameters.join(parameters)
        if original_n_params != self.n_parameters:
            self.set_fail_gradient()

    @property
    def name(self):
        return add_spaces(type(self).__name__)

    @property
    def n_parameters(self):
        return len(self._parameters)

    @property
    def has_separable_problem(self):
        return self._has_separable_problem

    @property
    def target(self):
        return self._target

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        self._parameters = parameters

    @property
    def transformation(self):
        return self._transformation

    @transformation.setter
    def transformation(self, transformation):
        self._transformation = transformation

    @property
    def pybamm_solution(self):
        return self.problem.pybamm_solution if self.problem is not None else None
