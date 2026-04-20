import numpy as np

from pybop.analysis.sensitivity_analysis import sensitivity_analysis
from pybop.costs.base_cost import BaseCost
from pybop.costs.evaluation import Evaluation
from pybop.parameters.parameter import Inputs, Parameters
from pybop.simulators.base_simulator import BaseSimulator, Solution
from pybop.simulators.failed_solution import FailedSolution


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
        self._parameters = Parameters()

        # Gather information from the simulator
        self._simulator = simulator.copy() if simulator is not None else BaseSimulator()
        self._has_sensitivities = self._simulator.has_sensitivities
        self.parameters.join(self._simulator.parameters)

        # Gather information from the cost function
        self._cost = cost or BaseCost()
        self._minimising = self._cost.minimising
        self.parameters.join(self._cost.parameters)

        # Update the simulator output variables to match the target
        self.set_target()

    def get_model_inputs(self, inputs: Inputs):
        return {key: inputs[key] for key in self._simulator.parameters.keys()}

    def __call__(self, inputs: Inputs | list[Inputs]) -> float | list[float]:
        """
        Evaluate the cost for one or more sets of inputs and return the cost value(s).

        Parameters
        ----------
        inputs : Inputs | list[Inputs]
            Input parameters for cost evaluation. Supports a list of inputs.

        Returns
        -------
        float | list[float]
            The cost value(s).
        """
        evaluation = self.evaluate(inputs=inputs, calculate_sensitivities=False)

        return (
            evaluation.values.item()
            if len(evaluation.values) == 1
            else evaluation.values.tolist()
        )

    def evaluate(
        self, inputs: Inputs | list[Inputs], calculate_sensitivities: bool = False
    ) -> Evaluation:
        """
        Evaluate the cost for one or more sets of inputs and return the cost value(s)
        and (optionally) the sensitivities.

        Parameters
        ----------
        inputs : Inputs | list[Inputs]
            Input parameters for cost evaluation. Supports a list of inputs.
        calculate_sensitivities : bool
            Whether to also return the sensitivities (default: False).

        Returns
        -------
        Evaluation
            The cost value(s) and (optionally) the gradient of the cost with respect to
            each input parameter.
        """
        # Accept numeric values, convert to Inputs dictionaries
        if not isinstance(inputs, dict):
            if not isinstance(inputs[0], dict):
                values = np.atleast_2d(inputs)
                inputs = [self.parameters.to_dict(v) for v in values]

        inputs_list = inputs if isinstance(inputs, list) else [inputs]

        return self.evaluate_batch(
            inputs=inputs_list, calculate_sensitivities=calculate_sensitivities
        )

    def evaluate_batch(
        self, inputs: list[Inputs], calculate_sensitivities: bool = False
    ) -> Evaluation:
        """
        Evaluate the cost for each set of inputs and return the cost value(s) and
        (optionally) the sensitivities.

        Parameters
        ----------
        inputs : list[Inputs]
            A list of input parameters.
        calculate_sensitivities : bool
            Whether to also return the sensitivities (default: False).

        Returns
        -------
        Evaluation
            Cost values of len(inputs) and (optionally) the gradient of the cost with respect to
            each input parameter with shape (len(inputs), len(parameters)).
        """
        if calculate_sensitivities:
            calculate_sensitivities = self._has_sensitivities

        solutions = self.simulate_batch(
            inputs=inputs, calculate_sensitivities=calculate_sensitivities
        )

        return self._cost.evaluate_batch(
            solutions, inputs=inputs, calculate_sensitivities=calculate_sensitivities
        )

    def simulate(
        self, inputs: Inputs | list[Inputs], calculate_sensitivities: bool = False
    ) -> Solution | list[Solution]:
        """
        Simulate the model for one or more sets of inputs and return the solution and
        (optionally) the sensitivities.

        Parameters
        ----------
        inputs : Inputs | list[Inputs]
            Input parameters. Support a list of inputs.
        calculate_sensitivities : bool
            Whether to also return the sensitivities (default: False).

        Returns
        -------
        Solution | list[Solution]
            The simulated model output y(t) and (optionally) the sensitivities dy/dx(t)
             for output variable(s) y, domain t and parameter(s) x.
        """
        if not isinstance(inputs, list):
            return self.simulate_batch(
                inputs=[inputs], calculate_sensitivities=calculate_sensitivities
            )[0]

        return self.simulate_batch(
            inputs=inputs, calculate_sensitivities=calculate_sensitivities
        )

    def simulate_batch(
        self, inputs: list[Inputs], calculate_sensitivities: bool = False
    ) -> list[Solution | FailedSolution]:
        """
        Simulate the model for each set of inputs and return the solution and
        (optionally) the sensitivities.

        Parameters
        ----------
        inputs : list[Inputs]
            A list of input parameters.

        Returns
        -------
        list[Solution]
            A list of length(inputs) containing the simulated model output y(t) and (optionally)
            the sensitivities dy/dx(t) for output variable(s) y, domain t and parameter(s) x.
        """
        # Check the validity of the inputs so we only evaluate valid parameters
        validity = []
        valid_inputs = []
        for x in inputs:
            if self.parameters.verify_inputs(x):
                validity.append(True)
                valid_inputs.append(x)
            else:
                validity.append(False)

        # Run simulations for the valid parameters
        model_inputs = [self.get_model_inputs(x) for x in valid_inputs]
        solutions = self._simulator.solve_batch(
            inputs=model_inputs, calculate_sensitivities=calculate_sensitivities
        )

        # Insert failed solutions for any invalid inputs
        invalid_indices = [i for i, valid in enumerate(validity) if not valid]
        for i in invalid_indices:
            solutions.insert(i, FailedSolution(self.target, inputs[i].keys()))

        return solutions

    def get_finite_initial_cost(self):
        """
        Compute the absolute initial cost, resampling the initial parameters if needed.
        """
        x0 = self.parameters.get_initial_values()
        cost0 = np.abs(self.evaluate(x0).values.item())
        nsamples = 0
        while np.isinf(cost0) and nsamples < 10:
            x0 = self.parameters.sample_from_distribution()[0]
            if x0 is None:
                break

            cost0 = np.abs(self.evaluate(x0).values.item())
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

    @property
    def cost(self):
        return self._cost

    @property
    def minimising(self):
        return self._minimising

    @property
    def domain(self):
        return self._cost.domain

    @property
    def domain_data(self):
        return self._cost.domain_data

    @property
    def target(self):
        return self._cost.target

    @property
    def target_data(self):
        return self._cost.target_data

    def set_target(self, value: list[str] | str | None = None):
        self._cost.set_target(value)
        self._simulator.set_output_variables(self._cost.target)

    @property
    def parameters(self):
        return self._parameters

    @property
    def n_parameters(self):
        return len(self._parameters)

    @property
    def simulator(self):
        return self._simulator

    @property
    def has_sensitivities(self):
        return self._has_sensitivities
