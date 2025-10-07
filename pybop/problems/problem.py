import numpy as np

from pybop.analysis.sensitivity_analysis import sensitivity_analysis
from pybop.costs.base_cost import BaseCost
from pybop.costs.evaluation import Evaluation
from pybop.costs.likelihoods import LogPosterior
from pybop.parameters.parameter import Inputs, Parameters
from pybop.simulators.base_simulator import BaseSimulator, Solution


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
            self._cost.log_likelihood.parameters = self.parameters
            self._cost.set_joint_prior()

    def get_model_inputs(self, inputs):
        all_values = list(inputs.values())
        n = len(self._simulator.parameters)
        return self._simulator.parameters.to_dict(all_values[:n])

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
            evaluation.values[0]
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
        # Convert values to parameter inputs
        if not isinstance(inputs, dict):
            if not isinstance(inputs[0], dict):
                values = np.atleast_2d(inputs)
                inputs = [self.parameters.to_dict(v) for v in values]
        inputs_list = inputs if isinstance(inputs, list) else [inputs]

        return self.batch_evaluate(
            inputs=inputs_list, calculate_sensitivities=calculate_sensitivities
        )

    def batch_evaluate(
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

        validity = []
        valid_inputs = []
        for x in inputs:
            # Check the validity of the inputs so we only evaluate valid parameters
            if self.parameters.verify_inputs(x):
                validity.append(True)
                valid_inputs.append(x)
            else:
                validity.append(False)

        # Run simulations for the valid parameters
        solutions = self.batch_simulate(
            valid_inputs, calculate_sensitivities=calculate_sensitivities
        )

        # Preallocate the evaluation results
        evaluation = Evaluation()
        evaluation.preallocate(
            n_inputs=len(inputs),
            n_parameters=len(self.parameters),
            calculate_sensitivities=calculate_sensitivities,
        )

        # Evaluate the cost for the valid parameters
        valid_indices = [i for i, valid in enumerate(validity) if valid]
        # TODO: Parallelise the cost computations
        if calculate_sensitivities:
            for i, sol in enumerate(solutions):
                e, de = self._cost.evaluate(
                    sol,
                    inputs=valid_inputs[i],
                    calculate_sensitivities=calculate_sensitivities,
                )
                evaluation.insert_result(i=valid_indices[i], value=e, sensitivities=de)
        else:
            for i, sol in enumerate(solutions):
                e = self._cost.evaluate(
                    sol,
                    inputs=valid_inputs[i],
                    calculate_sensitivities=calculate_sensitivities,
                )
                evaluation.insert_result(i=valid_indices[i], value=e)

        if False in validity:
            # Insert failure outputs for the invalid parameters into the lists of results
            invalid_indices = [i for i, valid in enumerate(validity) if not valid]
            if calculate_sensitivities:
                y, dy = self._cost.failure(calculate_sensitivities)
                for i in invalid_indices:
                    evaluation.insert_result(i=i, value=y, sensitivities=dy)
            else:
                y = self._cost.failure(calculate_sensitivities)
                for i in invalid_indices:
                    evaluation.insert_result(i=i, value=y)

        return evaluation

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
            return self.batch_simulate(
                inputs=[inputs], calculate_sensitivities=calculate_sensitivities
            )[0]

        return self.batch_simulate(
            inputs=inputs, calculate_sensitivities=calculate_sensitivities
        )

    def batch_simulate(
        self, inputs: list[Inputs], calculate_sensitivities: bool = False
    ) -> list[Solution]:
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
        model_inputs = [self.get_model_inputs(x) for x in inputs]
        return self._simulator.batch_solve(
            inputs=model_inputs, calculate_sensitivities=calculate_sensitivities
        )

    def get_finite_initial_cost(self):
        """
        Compute the absolute initial cost, resampling the initial parameters if needed.
        """
        x0 = self.parameters.get_initial_values()
        cost0 = np.abs(self.evaluate(x0).values[0])
        nsamples = 0
        while np.isinf(cost0) and nsamples < 10:
            x0 = self.parameters.sample_from_priors()
            if x0 is None:
                break

            cost0 = np.abs(self.evaluate(x0).values[0])
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
