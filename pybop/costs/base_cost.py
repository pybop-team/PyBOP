from collections.abc import Iterable

import numpy as np

from pybop._utils import add_spaces
from pybop.costs.evaluation import Evaluation
from pybop.parameters.parameter import Inputs, Parameters
from pybop.processing.dataset import Dataset
from pybop.simulators.solution import Solution


class BaseCost:
    """
    Base cost.

    Attributes
    ----------
    _de : float
        The gradient of the cost function to use if an error occurs during
        evaluation. Defaults to 1.0.
    minimising : bool, optional
        If False, tells the optimiser to switch the sign of the cost and gradient
        to maximise by default rather than minimise (default: True).
    """

    def __init__(self):
        self._de = 1.0
        self.parameters = Parameters()
        self.minimising = True
        # TODO: Remove the default domain, target and dataset from the base cost as
        # they are not relevant for all costs.
        self._domain = "Time [s]"
        self._target = ["Voltage [V]"]
        self._domain_data = None
        self._target_data = None
        self._dataset = None

    def set_target(
        self, target: list[str] | str | None = None, dataset: Dataset | None = None
    ):
        """Set the target variable and target data from a pybop.Dataset."""
        self._target = [target] if isinstance(target, str) else target or self._target
        self.n_outputs = len(self._target)

        if not isinstance(dataset, Dataset | None):
            raise ValueError("Dataset must be a pybop.Dataset object.")

        if dataset is not None:
            # Check that the dataset contains necessary variables
            dataset.check(domain=dataset.domain, signal=self._target)
            self._domain = dataset.domain
            self._dataset = dataset.data

        if self._dataset is not None:
            # Unpack the domain and target data
            self._domain_data = self._dataset[self.domain]
            self.n_data = len(self._domain_data)
            self._target_data = {var: self._dataset[var] for var in self._target}

    def evaluate(
        self,
        solution: Solution | list[Solution],
        inputs: Inputs | list[Inputs] | None = None,
        calculate_sensitivities: bool = False,
    ) -> Evaluation:
        """
        Computes the cost function for the given predictions.

        Parameters
        ----------
        solution : Solution | list[Solution]
            The simulation result or a list of results.
        inputs : Inputs | list[Inputs], optional
            Input parameters or a list of inputs (default: None).
        calculate_sensitivities : bool
            Whether to also return the sensitivities (default: False).
        """
        solution_list = solution if isinstance(solution, list) else [solution]

        inputs = inputs or self.parameters.to_dict("initial")
        inputs_list = inputs if isinstance(inputs, list) else [inputs]
        if len(inputs_list) != len(solution_list):
            raise ValueError(
                f"The number of solutions ({len(solution_list)}) must match "
                f"the number of inputs ({len(inputs_list)})."
            )

        return self.evaluate_batch(
            solution=solution_list,
            inputs=inputs_list,
            calculate_sensitivities=calculate_sensitivities,
        )

    def evaluate_batch(
        self,
        solution: list[Solution],
        inputs: list[Inputs],
        calculate_sensitivities: bool = False,
    ) -> Evaluation:
        """
        Computes the cost function for the given predictions.

        Parameters
        ----------
        solution : list[Solution]
            A list of simulation results.
        inputs : list[Inputs]
            The corresponding list of input parameters.
        calculate_sensitivities : bool
            Whether to also return the sensitivities (default: False).
        """
        raise NotImplementedError

    def stack_sensitivities(self, solution: Solution) -> dict[str, np.ndarray]:
        """
        Stack the sensitivities for each output variable into a single array.

        Parameters
        ----------
        solution : pybop.Solution | pybamm.Solution
            A solution object containing a dictionary of sensitivities for each output variable.

        Returns
        -------
        dict[str, np.ndarray[np.float64]]
            The sensitivities dy/dx(t) for each output variable y with respect to each parameter
            x over the domain t. The dictionary keys are the parameter names and the arrays are
            of dimensions (len(target), len(domain_data)).
        """
        return {
            key: np.vstack([solution[var].sensitivities[key] for var in self.target])
            for key in solution.all_inputs[0].keys()
        }

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

    def failure(
        self, parameter_names: Iterable[str], calculate_sensitivities: bool = True
    ):
        sign = 1.0 if self.minimising else -1.0
        if calculate_sensitivities:
            return (sign * np.inf, {key: sign * self._de for key in parameter_names})
        return sign * np.inf

    @property
    def name(self):
        return add_spaces(type(self).__name__)

    @property
    def n_parameters(self):
        return len(self.parameters)

    @property
    def domain(self):
        return self._domain

    @property
    def domain_data(self):
        return self._domain_data

    @property
    def target(self):
        return self._target

    @property
    def target_data(self):
        return self._target_data


class LogPrior(BaseCost):
    """
    The log-prior as a cost, to be used with the LogPosterior class.
    """

    def __init__(self, parameters):
        super().__init__()
        self.parameters = parameters
        self.minimising = False

    def evaluate_batch(
        self,
        solution: list[Solution],
        inputs: list[Inputs],
        calculate_sensitivities: bool = False,
    ) -> Evaluation:
        """
        Computes the log-prior for the given inputs, and optionally the sensitivities.

        Parameters
        ----------
        solution : list[Solution]
            A list of simulation results.
        inputs : list[Inputs]
            The corresponding list of input parameters.
        calculate_sensitivities : bool
            Whether to also return the sensitivities (default: False).
        """
        l = np.empty(len(solution))
        dl = (
            {key: np.zeros(len(solution)) for key in inputs[0].keys()}
            if calculate_sensitivities
            else None
        )

        for i, x in enumerate(inputs):
            # Get the values of all input parameters
            input_values = np.asarray(list(x.values()))

            # Compute log prior (and gradient)
            if calculate_sensitivities:
                l[i], dl_array = self.parameters.distribution.logpdfS1(input_values)
                dl_i = {key: dl_array[i_key] for i_key, key in enumerate(x.keys())}

                # Use failure gradient for any infinite log prior
                if not np.isfinite(l[i]):
                    l[i], dl_i = self.failure(x.keys(), calculate_sensitivities)

                for key in x.keys():
                    dl[key][i] = dl_i[key]

            else:
                l[i] = self.parameters.distribution.logpdf(input_values)

        return Evaluation(values=l, sensitivities=dl)
