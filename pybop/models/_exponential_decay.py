import pybamm
from pybamm.models.base_model import BaseModel


class ExponentialDecayModel(BaseModel):
    """
    Exponential decay model defined by the equation:

        dy/dt = -k * y,  y(0) = y0

    Note: The output variables are named "y_{i}" for each state.
        For example, the first state is "y_0", the second is "y_1", etc.
    Attributes:
        n_states (int): Number of states in the system (default is 1).
        pybamm_model (pybamm.BaseModel): PyBaMM model representation.
        default_parameter_values (pybamm.ParameterValues): Default parameter values
            for the model, with "k" (decay rate) and "y0" (initial condition).

    Parameters:
        name (str): Name of the model (default: "Experimental Decay Model").
        parameter_set (pybamm.ParameterValues): Parameter values for the model.
        n_states (int): Number of states in the system. Must be >= 1.
    """

    def __init__(
        self,
        name: str = "Experimental Decay Model",
        n_states: int = 1,
    ):
        if n_states < 1:
            raise ValueError("The number of states (n_states) must be at least 1.")
        super().__init__(name=name)
        self.n_states = n_states

        # Initialise the PyBaMM model, variables, parameters
        ys = [pybamm.Variable(f"y_{i}") for i in range(n_states)]
        k = pybamm.Parameter("k")
        y0 = pybamm.Parameter("y0")

        # Set up the right-hand side and initial conditions
        self.rhs = {y: -k * y for y in ys}
        self.initial_conditions = {y: y0 for y in ys}

        # Define model outputs and set default values
        self.variables = {f"y_{en}": i for en, i in enumerate(ys)} | {
            "Time [s]": pybamm.t
        }

    @property
    def default_parameter_values(self):
        return pybamm.ParameterValues({"k": 0.1, "y0": 1})
