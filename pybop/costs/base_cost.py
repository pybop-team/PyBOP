import numpy as np

from pybop._utils import add_spaces
from pybop.parameters.parameter import Inputs, Parameters
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

        # Default settings, to be overwritten
        self.domain = "Time [s]"
        self.target = ["Voltage [V]"]
        self._domain_data = None
        self._target_data = None

    def evaluate(
        self,
        solution: Solution,
        inputs: Inputs | None = None,
        calculate_sensitivities: bool = False,
    ) -> float | tuple[float, np.ndarray]:
        """
        Computes the cost function for the given predictions.

        Parameters
        ----------
        solution : pybop.Solution | pybamm.Solution
            The simulation result.
        inputs : Inputs, optional
            Input parameters (default: None).
        calculate_sensitivities : bool
            Whether to also return the sensitivities (default: False).

        Returns
        -------
        np.float64 or tuple[np.float64, np.ndarray[np.float64]]
            If the solution has sensitivities, returns a tuple containing the cost (float) and the
            gradient with dimension (len(parameters)), otherwise returns only the cost.
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

    def failure(self, parameter_names: list[str], calculate_sensitivities: bool = True):
        if calculate_sensitivities:
            return (
                (np.inf, {key: self._de for key in parameter_names})
                if self.minimising
                else (-np.inf, {key: -self._de for key in parameter_names})
            )
        else:
            return np.inf if self.minimising else -np.inf

    @property
    def name(self):
        return add_spaces(type(self).__name__)

    @property
    def n_parameters(self):
        return len(self.parameters)

    @property
    def domain_data(self):
        return self._domain_data

    @property
    def target_data(self):
        return self._target_data
