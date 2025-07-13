import warnings

import casadi
import numpy as np
import pybamm
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

from pybop import Parameters, SymbolReplacer
from pybop.pipelines._pybamm_pipeline import PybammPipeline


class PybammEISPipeline:
    """
    A class to build an EIS PyBaMM pipeline for a given model and experiment, and run the resultant
    simulation.

    There are two contexts in which this class can be used:
    1. build_on_eval=True: A pybamm model needs to be built multiple times with different parameter
        values, for the case where any parameters is a geometric parameter, which changes the mesh.
    2. build_on_eval=False: A pybamm model needs to be built once, and then run multiple times with
        different input parameters.
    """

    def __init__(
        self,
        model: pybamm.BaseModel,
        f_eval: np.ndarray | list[float],
        geometry: pybamm.Geometry | None = None,
        parameter_values: pybamm.ParameterValues | None = None,
        submesh_types: dict | None = None,
        var_pts: dict | None = None,
        spatial_methods: dict | None = None,
        solver: pybamm.BaseSolver | None = None,
        pybop_parameters: Parameters | None = None,
        initial_state: float | str | None = None,
        build_on_eval: bool = True,
    ):
        """
        Parameters
        ---------
        model : pybamm.BaseModel
            The PyBaMM model to be used.
        f_eval : list
            The frequencies at which to evaluate the impedance.
        geometry: pybamm.Geometry (optional)
            The geometry upon which to solve the model.
        parameter_values : pybamm.ParameterValues (optional)
            Parameters and their corresponding numerical values.
        submesh_types : dict (optional)
            A dictionary of the types of submesh to use on each subdomain.
        var_pts : dict (optional)
            A dictionary of the number of points used by each spatial variable.
        spatial_methods : dict (optional)
            A dictionary of the types of spatial method to use on each.
            domain (e.g. pybamm.FiniteVolume)
        solver : pybamm.BaseSolver (optional)
            The solver to use to solve the model.
        pybop_parameters : pybop.Parameters (optional)
            The parameters to be optimised.
        initial_state: float | str (optional)
            The initial state of charge or voltage for the battery model. If float, it will be
            represented as SoC and must be in range 0 to 1. If str, it will be represented as voltage and
            needs to be in the format: "3.4 V".
        build_on_eval : bool
            Boolean to determine if the model will be rebuilt every evaluation (default: True).
        """

        self._pybamm_pipeline = PybammPipeline(
            model,
            geometry=geometry,
            parameter_values=parameter_values,
            submesh_types=submesh_types,
            var_pts=var_pts,
            spatial_methods=spatial_methods,
            solver=solver,
            pybop_parameters=pybop_parameters,
            initial_state=initial_state,
            build_on_eval=build_on_eval,
        )

        # Set-up model for EIS
        self._f_eval = f_eval
        self.set_up_for_eis(self._pybamm_pipeline.model)
        self._pybamm_pipeline.parameter_values["Current function [A]"] = 0

        # Initialise
        self.M = None
        self._jac = None
        self.b = None

        v_scale = getattr(model.variables["Voltage [V]"], "scale", 1)
        i_scale = getattr(model.variables["Current [A]"], "scale", 1)
        self.z_scale = self._pybamm_pipeline.parameter_values.evaluate(
            v_scale / i_scale
        )

    def set_up_for_eis(self, model: pybamm.BaseModel) -> None:
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
            None,
            model.options,
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

    def initialise_eis_pipeline(self):
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
        inputs = self._pybamm_pipeline.pybop_parameters.to_dict()
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
