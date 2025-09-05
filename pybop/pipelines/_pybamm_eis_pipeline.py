import warnings

import casadi
import numpy as np
import pybamm
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

from pybop import Inputs, Parameters, SymbolReplacer
from pybop.pipelines._pybamm_pipeline import PybammPipeline


class PybammEISPipeline:
    """
    A class to build an EIS PyBaMM pipeline for a given model and data, and run the resultant simulation.

    There are two contexts in which this class can be used:
    1. A pybamm model needs to be built once, and then run multiple times with different input parameters
    2. A pybamm model needs to be built multiple times with different parameter values,
        for the case where some of the parameters are geometric parameters which change the mesh

    The logic for (1) and (2) occurs within the composed PybammPipeline and happens automatically.
    To override this logic, the argument `build_on_eval` can be set to `True` which will force (2) to
    occur.
    """

    def __init__(
        self,
        model: pybamm.BaseModel,
        f_eval: np.ndarray | list[float],
        pybop_parameters: Parameters | None = None,
        parameter_values: pybamm.ParameterValues | None = None,
        initial_state: float | str | None = None,
        solver: pybamm.BaseSolver | None = None,
        geometry: pybamm.Geometry | None = None,
        submesh_types: dict | None = None,
        var_pts: dict | None = None,
        spatial_methods: dict | None = None,
        discretisation_kwargs: dict | None = None,
        build_on_eval: bool = False,
    ):
        """
        Parameters
        ----------
        model : pybamm.BaseModel
            The PyBaMM model to be used.
        f_eval : list
            The frequencies at which to evaluate the impedance.
        pybop_parameters : pybop.Parameters
            The parameters to optimise.
        parameter_values : pybamm.ParameterValues
            The parameters to be used in the model.
        initial_state : float | str
            The initial state of charge or voltage for the battery model. If float, it will be represented
            as SoC and must be in range 0 to 1. If str, it will be represented as voltage and needs to be in
            the format: "3.4 V".
        solver : pybamm.BaseSolver
            The solver to be used. If None, the idaklu solver will be used.
        geometry : pybamm.Geometry, optional
            The geometry upon which to solve the model.
        submesh_types : dict, optional
            A dictionary of the types of submesh to use on each subdomain.
        var_pts : dict, optional
            A dictionary of the number of points used by each spatial variable.
        spatial_methods : dict, optional
            A dictionary of the types of spatial method to use on each domain (e.g. pybamm.FiniteVolume).
        discretisation_kwargs : dict, optional
            Any keyword arguments to pass to the Discretisation class.
            See :class:`pybamm.Discretisation` for details.
        build_on_eval : bool, optional
            If True, the model will be rebuilt every evaluation. Otherwise, the need to rebuild will be
            determined automatically.
        """

        # Set-up model for EIS
        self._f_eval = f_eval
        model = self.set_up_for_eis(model)
        parameter_values = parameter_values or model.default_parameter_values
        parameter_values["Current function [A]"] = 0

        # Create PyBaMM pipeline
        self._pybamm_pipeline = PybammPipeline(
            model,
            pybop_parameters=pybop_parameters,
            parameter_values=parameter_values,
            initial_state=initial_state,
            solver=solver,
            geometry=geometry,
            submesh_types=submesh_types,
            var_pts=var_pts,
            spatial_methods=spatial_methods,
            discretisation_kwargs=discretisation_kwargs,
            build_on_eval=build_on_eval,
        )

        # Initialise
        self.M = None
        self._jac = None
        self.b = None

        v_scale = getattr(model.variables["Voltage [V]"], "scale", 1)
        i_scale = getattr(model.variables["Current [A]"], "scale", 1)
        self.z_scale = self._pybamm_pipeline.parameter_values.evaluate(
            v_scale / i_scale
        )

    def set_up_for_eis(self, model: pybamm.BaseModel) -> pybamm.BaseModel:
        """
        Set up the model for electrochemical impedance spectroscopy (EIS) simulations.
        This method adds the necessary algebraic equations and variables to the model.
        Originally developed by pybamm-eis: https://github.com/pybamm-team/pybamm-eis

        Parameters
        ----------
        model : pybamm.BaseModel
            The PyBaMM model to be used for EIS simulations.

        Returns
        -------
        pybamm.BaseModel
            The modified model ready for EIS simulations.

        Raises
        ------
        ValueError
            If the model is missing required variables.
        """
        # Verify model has required variables
        required_vars = ["Voltage [V]", "Current [A]"]
        for var in required_vars:
            if var not in model.variables:
                raise ValueError(
                    f"Model must contain variable '{var}' for EIS simulation"
                )

        V_cell = pybamm.Variable("Voltage variable [V]")
        model.variables["Voltage variable [V]"] = V_cell
        V = model.variables["Voltage [V]"]

        # Add algebraic equation for the voltage
        model.algebraic[V_cell] = V_cell - V
        model.initial_conditions[V_cell] = model.param.ocv_init

        # Create the FunctionControl submodel and extract variables
        external_circuit_variables = pybamm.external_circuit.FunctionControl(
            model.param,
            external_circuit_function=None,
            options=model.options,
            control="algebraic",
        ).get_fundamental_variables()

        # Define the variables to replace
        symbol_replacement_map = {}
        for name, variable in external_circuit_variables.items():
            if name in model.variables.keys():
                symbol_replacement_map[model.variables[name]] = variable

        # Don't replace initial conditions, as these should not contain
        # variable objects
        replacer = SymbolReplacer(
            symbol_replacement_map, process_initial_conditions=False
        )
        replacer.process_model(model, inplace=True)

        # Add an algebraic equation for the current density variable
        # External circuit submodels are always equations on the current
        I_cell = model.variables["Current variable [A]"]
        I = model.variables["Current [A]"]
        I_applied = pybamm.FunctionParameter(
            "Current function [A]", {"Time [s]": pybamm.t}
        )
        model.algebraic[I_cell] = I - I_applied
        model.initial_conditions[I_cell] = 0

        return model

    def initialise_eis_pipeline(self, inputs: Inputs) -> None:
        """
        Initialise the electrochemical impedance spectroscopy (EIS) simulation.
        This method sets up the mass matrix and solver, converts inputs to the appropriate format,
        extracts the necessary attributes from the model, and prepares matrices for the simulation.

        Raises
        ------
        RuntimeError
            If the model hasn't been built yet.
        """
        built_model = self._pybamm_pipeline.built_model
        M = self._pybamm_pipeline.built_model.mass_matrix.entries
        self._pybamm_pipeline.solver.set_up(built_model, inputs=inputs)

        # Convert inputs to casadi format if needed
        casadi_inputs = (
            casadi.vertcat(*inputs.values())
            if inputs is not None and built_model.convert_to_format == "casadi"
            else inputs or []
        )

        # Extract the necessary attributes from the model
        y0 = built_model.concatenated_initial_conditions.evaluate(0, inputs=inputs)
        jac = built_model.jac_rhs_algebraic_eval(0, y0, casadi_inputs).sparse()

        # Convert to Compressed Sparse Column format
        self.M = csc_matrix(M)
        self._jac = csc_matrix(jac)

        # Add forcing to the RHS on the current density
        self.b = np.zeros(y0.shape)
        self.b[-1] = -1

    def calculate_impedance(self, frequency):
        """
        Calculate the impedance for a given frequency.

        This method computes the system matrix, solves the linear system, and calculates
        the impedance based on the solution.

        Parameters
        ----------
        frequency : float
            The frequency at which to calculate the impedance in Hz.

        Returns
        -------
        complex
            The calculated impedance.
        """

        # Compute the system matrix
        A = 1.0j * 2 * np.pi * frequency * self.M - self._jac

        # Solve the system
        x = spsolve(A, self.b)

        # Calculate the impedance (voltage / current)
        return -x[-2] / x[-1]

    def solve(self, calculate_sensitivities: bool = False) -> np.ndarray:
        """
        Run the EIS simulation to calculate impedance at all specified frequencies.

        Parameters
        ---------
        calculate_sensitivities : bool, optional
            Whether to calculate sensitivities or not. Default is False.
            Currently not implemented for EIS.

        Returns
        -------
        np.ndarray
            Complex array containing the impedance values with corresponding frequencies.
        """
        if calculate_sensitivities:
            warnings.warn(
                "Sensitivity calculation not implemented for EIS simulations",
                stacklevel=2,
            )

        zs = [self.calculate_impedance(frequency) for frequency in self._f_eval]

        return np.asarray(zs) * self.z_scale

    @property
    def pybamm_pipeline(self):
        return self._pybamm_pipeline

    @property
    def parameter_values(self):
        return self._pybamm_pipeline.parameter_values

    @property
    def pybop_parameters(self):
        return self._pybamm_pipeline.pybop_parameters

    @property
    def requires_rebuild(self):
        return self._pybamm_pipeline.requires_rebuild

    def rebuild(self, params: Inputs) -> None:
        """Update the parameter values and rebuild the PyBaMM EIS pipeline."""
        self._pybamm_pipeline.rebuild(params)

    def build(self) -> None:
        """Build the PyBaMM EIS pipeline."""
        self._pybamm_pipeline.build()
