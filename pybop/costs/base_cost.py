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
    minimising : bool, optional
        If False, tells the optimiser to switch the sign of the cost and gradient
        to maximise by default rather than minimise (default: True).
    """

    class DeferredPrediction:
        """
        Class used to indicate a prediction has yet to be, but is expected during
        a cost call.
        """

        pass

    def __init__(self, problem: BaseProblem | None = None):
        self._parameters = Parameters()
        self.problem = problem
        self.verbose = False
        self._has_separable_problem = False
        self.y = None
        self.dy = None
        self._de = 1.0
        self.minimising = True
        self.has_sensitivities = False
        if isinstance(self.problem, BaseProblem):
            self._target = self.problem.target
            self._parameters.join(self.problem.parameters)
            self.n_outputs = self.problem.n_outputs
            self.output_variables = self.problem.output_variables
            self._has_separable_problem = True
            self.grad_fail = None
            self.set_fail_gradient()
            self.has_sensitivities = self.problem.has_sensitivities

    def __call__(
        self,
        inputs: Inputs | list | np.ndarray,
        calculate_grad: bool = False,
    ) -> (
        float | tuple[float, np.ndarray] | list[float] | list[tuple[float, np.ndarray]]
    ):
        """
        Compute cost and optional gradient for given input parameters.

        Parameters
        ----------
        inputs : Inputs | list[Inputs] | list[float] | np.adarray
            Input parameters for cost computation. Supports list-like evaluation of
            multiple input values, shaped [N,M] where N is the number of input positions
            to evaluate and M is the number of inputs for the underlying model (i.e. parameters).
        calculate_grad : bool
            If True, the gradient will be computed as well as the cost (default: False).

        Returns
        -------
        Union[float, list, tuple[float, np.ndarray], list[tuple[float, np.ndarray]]]
            - Single input, no gradient: float
            - Multiple inputs, no gradient: list[float]
            - Single input with gradient: tuple[float, np.ndarray]
            - Multiple inputs with gradient: list[tuple[float, np.ndarray]]
        """
        # Convert values to parameter inputs
        if not isinstance(inputs, dict):
            if not isinstance(inputs[0], dict):
                values = np.atleast_2d(inputs)
                inputs = [self.parameters.to_dict(v) for v in values]
        inputs_list = inputs if isinstance(inputs, list) else [inputs]

        results = []
        for inputs in inputs_list:
            result = self.single_call(inputs, calculate_grad=calculate_grad)
            results.append(result)

        return results[0] if len(inputs_list) == 1 else results

    def single_call(
        self,
        inputs: Inputs,
        calculate_grad: bool,
    ) -> float | tuple[float, np.ndarray]:
        """Evaluate the cost and (optionally) the gradient for a single set of inputs."""
        if calculate_grad:
            calculate_grad = self.has_sensitivities

        # Check the validity of the parameters before evaluating the cost
        if not self.parameters.verify_inputs(inputs):
            if calculate_grad:
                return (
                    (np.inf, self.grad_fail)
                    if self.minimising
                    else (-np.inf, -self.grad_fail)
                )
            else:
                return np.inf if self.minimising else -np.inf

        self.parameters.update(values=list(inputs.values()))

        y = self.DeferredPrediction
        dy = self.DeferredPrediction if calculate_grad else None

        if self._has_separable_problem:
            if calculate_grad:
                y, dy = self.problem.evaluateS1(self.problem.parameters.to_dict())
                return self.compute(y, dy=dy)

            y = self.problem.evaluate(self.problem.parameters.to_dict())

        return self.compute(y, dy=dy)

    def compute(self, y: dict, dy: np.ndarray | None):
        """
        Compute the cost and, if dy is not None, its gradient with respect to the
        parameters.

        This method only computes the cost, without calling the `problem.evaluate()`.
        This method must be implemented by subclasses.

        Parameters
        ----------
        y : dict
            A dictionary of predictions with keys designating the output variables for fitting.
        dy : np.ndarray, optional
            The corresponding gradient with respect to the parameters for each output variable.

        Raises
        ------
        NotImplementedError
            If the method has not been implemented by the subclass.
        """
        raise NotImplementedError

    def get_finite_initial_cost(self):
        """
        Compute the absolute initial cost, resampling the initial parameters if needed.
        """
        x0 = self.parameters.get_initial_values()
        cost0 = np.abs(self.__call__(x0))
        nsamples = 0
        while np.isinf(cost0) and nsamples < 10:
            x0 = self.parameters.sample_from_priors()
            if x0 is None:
                break

            cost0 = np.abs(self.__call__(x0))
            nsamples += 1
        if nsamples > 0:
            self.parameters.update(initial_values=x0)

        if np.isinf(cost0):
            raise ValueError("The initial parameter values return an infinite cost.")
        return cost0

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
            "names": list(self.parameters.keys()),
            "bounds": self.parameters.get_bounds_array(),
            "num_vars": len(self.parameters),
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
            A dictionary of predictions with keys designating the output variables for fitting.

        Returns
        -------
        bool
            True if the prediction matches the target data, otherwise False.
        """
        if any(
            len(y.get(key, [])) != len(self._target.get(key, []))
            for key in self.output_variables
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

    def stack_sensitivities(self, dy) -> np.ndarray:
        """
        Stack the sensitivities for each output variable and parameter into a single array.

        Parameters
        ----------
        dict[str, dict[str, np.ndarray[np.float64]]]
            A dictionary of the sensitivities dy/dx(t) for each parameter x and output_variables y.

        Returns
        -------
        np.ndarray[np.float64]
            The combined sensitivities dy/dx(t) for each parameter and output_variables, with
            dimensions of (len(parameters), len(output_variables), len(domain_data)).
        """
        return np.stack(
            [
                np.row_stack([dy[key][var] for var in self.output_variables])
                for key in dy.keys()
            ],
            axis=0,
        )

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
