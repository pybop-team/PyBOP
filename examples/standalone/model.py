import pybamm

from pybop.models.base_model import BaseModel


class ExponentialDecay(BaseModel):
    """
    Exponential decay model with two parameters y0 and k

    dy/dt = -ky
    y(0) = y0

    """

    def __init__(
        self,
        name: str = "Constant Model",
        parameter_set: pybamm.ParameterValues = None,
        n_states: int = 1,
    ):
        super().__init__(name=name, parameter_set=parameter_set)

        self.n_states = n_states
        if n_states < 1:
            raise ValueError("The number of states (n_states) must be greater than 0")
        self.pybamm_model = pybamm.BaseModel()
        ys = [pybamm.Variable(f"y_{i}") for i in range(n_states)]
        k = pybamm.Parameter("k")
        y0 = pybamm.Parameter("y0")
        self.pybamm_model.rhs = {y: -k * y for y in ys}
        self.pybamm_model.initial_conditions = {y: y0 for y in ys}
        self.pybamm_model.variables = {"y_0": ys[0], "2y": 2 * ys[0]}

        default_parameter_values = pybamm.ParameterValues(
            {
                "k": 0.1,
                "y0": 1,
            }
        )

        self._unprocessed_model = self.pybamm_model

        self.default_parameter_values = (
            default_parameter_values
            if self._parameter_set is None
            else self._parameter_set
        )
        self._parameter_set = self.default_parameter_values
        self._unprocessed_parameter_set = self._parameter_set

        self.geometry = self.pybamm_model.default_geometry
        self.submesh_types = self.pybamm_model.default_submesh_types
        self.var_pts = self.pybamm_model.default_var_pts
        self.spatial_methods = self.pybamm_model.default_spatial_methods
        self.solver = pybamm.CasadiSolver(mode="fast")
        self._model_with_set_params = None
        self._built_model = None
        self._built_initial_soc = None
        self._mesh = None
        self._disc = None
        self.geometric_parameters = {}
