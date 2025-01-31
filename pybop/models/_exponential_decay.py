import pybamm

from pybop.models.base_model import BaseModel


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
        parameter_set: pybamm.ParameterValues = None,
        n_states: int = 1,
        solver=None,
    ):
        if n_states < 1:
            raise ValueError("The number of states (n_states) must be at least 1.")

        super().__init__(name=name, parameter_set=parameter_set)

        self.n_states = n_states
        if solver is None:
            self._solver = pybamm.CasadiSolver
            self._solver.mode = "fast with events"
            self._solver.max_step_decrease_count = 1
        else:
            self._solver = solver

        # Initialise the PyBaMM model, variables, parameters
        self.pybamm_model = pybamm.BaseModel()
        ys = [pybamm.Variable(f"y_{i}") for i in range(n_states)]
        k = pybamm.Parameter("k")
        y0 = pybamm.Parameter("y0")

        # Set up the right-hand side and initial conditions
        self.pybamm_model.rhs = {y: -k * y for y in ys}
        self.pybamm_model.initial_conditions = {y: y0 for y in ys}

        # Define model outputs and set default values
        self.pybamm_model.variables = {f"y_{en}": i for en, i in enumerate(ys)} | {
            "Time [s]": pybamm.t
        }
        self.default_parameter_values = parameter_set or pybamm.ParameterValues(
            {"k": 0.1, "y0": 1}
        )

        # Store model attributes to be used by the solver
        self._unprocessed_model = self.pybamm_model
        self._parameter_set = self.default_parameter_values
        self._unprocessed_parameter_set = self._parameter_set

        # Geometry and solver setup
        self._geometry = self.pybamm_model.default_geometry
        self._submesh_types = self.pybamm_model.default_submesh_types
        self._var_pts = self.pybamm_model.default_var_pts
        self._spatial_methods = self.pybamm_model.default_spatial_methods
        self._solver = pybamm.CasadiSolver(mode="fast")

        # Additional attributes for solver and discretisation
        self._model_with_set_params = None
        self._built_model = None
        self._built_initial_soc = None
        self._mesh = None
        self._disc = None
        self.geometric_parameters = {}
