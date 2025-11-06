import numpy as np

from pybop.analysis.sensitivity_analysis import sensitivity_analysis
from pybop.costs.base_cost import BaseCost
from pybop.costs.likelihoods import LogPosterior
from pybop.parameters.parameter import Inputs, Parameters
from pybop.simulators.base_simulator import BaseSimulator


class Problem:
    """
    Base class for defining a problem within the PyBOP framework, compatible with PINTS.

    Parameters
    ----------
    simulator : pybop.BaseSimulator
        The model, protocol and optional dataset combined into a simulator object.
    parameters : list[pybop.Parameter] or pybop.Parameters
        An object or list of the parameters for the problem.
    cost : pybop.BaseCost, optional
        An cost e.g. an error measure, a log-likelihood or a design cost.

    Attributes
    ----------
    target_data : array-like
        An array containing the target data to fit.
    n_outputs : int
        The number of outputs in the model.
    minimising : bool, optional
        If False, tells the optimiser to switch the sign of the cost and gradient
        to maximise by default rather than minimise (default: True).
    """

    def __init__(self, simulator: BaseSimulator = None, cost: BaseCost = None):
        self.parameters = Parameters()

        # Gather information from the simulator
        self._simulator = simulator.copy() if simulator is not None else BaseSimulator()
        self._has_sensitivities = self._simulator.has_sensitivities
        self.parameters.join(self._simulator.parameters)

        # Gather information from the cost function
        self._cost = cost or BaseCost()
        self._minimising = self._cost.minimising
        self.domain = self._cost.domain
        self.target = self._cost.target

        # Share parameters from model with cost
        self.parameters.join(self._cost.parameters)
        self._cost.parameters = self.parameters
        self._cost.set_fail_gradient()

        # Objective-specific configuration
        if isinstance(self._cost, LogPosterior):
            self._cost.set_joint_prior()

        # Reset parameters from both model and cost
        self.parameters.reset_to_initial()

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, value: list[str] | None):
        self._target = [value] if isinstance(value, str) else value or []
        self._simulator.set_output_variables(self._target)

    @property
    def n_outputs(self):
        return len(self._target)

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
            calculate_grad = self._has_sensitivities

        # Check the validity of the parameters before evaluating the cost
        if not self.parameters.verify_inputs(inputs):
            return self._cost.failure(calculate_grad if calculate_grad else None)

        self.parameters.update(values=list(inputs.values()))

        if calculate_grad:
            y, dy = self.simulateS1(self._simulator.parameters.to_dict())
            return self._cost.compute(y, dy=dy)

        y = self.simulate(self._simulator.parameters.to_dict())
        return self._cost.compute(y, dy=None)

    def batch_call(
        self, inputs_list: list[Inputs], calculate_grad: bool
    ) -> list[float] | list[tuple[float, np.ndarray]]:
        """Evaluate the cost and (optionally) the gradient for a list of inputs."""

        # TODO: Upgrade the cost evaluations for batch processing

        if calculate_grad:
            costs, grads = [], []
            for inputs in inputs_list:
                out = self.single_call(inputs, calculate_grad=True)
                costs.append(out[0])
                grads.append(out[1])
            return np.asarray(costs), np.asarray(grads)

        costs = []
        for inputs in inputs_list:
            costs.append(self.single_call(inputs, calculate_grad=False))
        return np.atleast_1d(costs)

    def simulate(
        self, inputs: Inputs
    ) -> (
        dict[str, np.ndarray]
        | tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]
    ):
        """
        Evaluate the model with the given parameters and return the target.

        Parameters
        ----------
        inputs : Inputs
            Parameters for evaluation of the model.

        Returns
        -------
        dict[str, np.ndarray[np.float64]]
            The simulated model output y(t), or y(ω) for EIS, for the given inputs.
        """
        return self._simulator.simulate(inputs=inputs, calculate_sensitivities=False)

    def simulateS1(self, inputs: Inputs):
        """
        Evaluate the model with the given parameters and return the target and
        their derivatives.

        Parameters
        ----------
        inputs : Inputs
            Parameters for evaluation of the model.

        Returns
        -------
        tuple[dict[str, np.ndarray[np.float64]], dict[str, dict[str, np.ndarray]]]
            A tuple containing the simulation result y(t) and the sensitivities dy/dx(t)
            for each parameter x and output variables y simulated with the given inputs.
        """
        return self._simulator.simulate(inputs=inputs, calculate_sensitivities=True)

    def get_finite_initial_cost(self):
        """
        Compute the absolute initial cost, resampling the initial parameters if needed.
        """
        x0 = self.parameters.get_initial_values()
        cost0 = np.abs(self.__call__(x0))
        nsamples = 0
        while np.isinf(cost0) and nsamples < 10:
            x0 = self.parameters.sample_from_distributions()
            if x0 is None:
                break

            cost0 = np.abs(self.__call__(x0))
            nsamples += 1
        if nsamples > 0:
            self.parameters.update(initial_values=x0)

        if np.isinf(cost0):
            raise ValueError("The initial parameter values return an infinite cost.")
        return cost0

    def sensitivity_analysis(
        self, n_samples: int = 256, calc_second_order: bool = False
    ) -> dict:
        """
        Computes the parameter sensitivities on the combined cost function using
        SOBOL analysis. See pybop.analysis.sensitivity_analysis for more details.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples for SOBOL sensitivity analysis, performs best as a
            power of 2, i.e. 128, 256, etc.
        calc_second_order : bool, optional
            Whether to calculate second-order sensitivities.
        """
        return sensitivity_analysis(
            problem=self, n_samples=n_samples, calc_second_order=calc_second_order
        )

    def join_parameters(self, parameters):
        """
        Setter for joining parameters. This method sets the fail gradient if the join adds parameters.
        """
        original_n_params = self.n_parameters
        self._parameters.join(parameters)
        if original_n_params != self.n_parameters:
            self.set_fail_gradient()

    @property
    def cost(self):
        return self._cost

    @property
    def minimising(self):
        return self._minimising

    @property
    def target_data(self):
        return self._cost.target_data

    @property
    def domain_data(self):
        return self._cost.domain_data

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        self._parameters = parameters

    @property
    def n_parameters(self):
        return len(self._parameters)

    @property
    def simulator(self):
        return self._simulator

    @property
    def has_sensitivities(self):
        return self._has_sensitivities

    @property
    def eis(self):
        return self._eis
