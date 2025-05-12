import casadi
import numpy as np
import pybamm
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

from pybop import Parameters, SymbolReplacer
from pybop._pybamm_pipeline import PybammPipeline


class PybammEISPipeline(PybammPipeline):
    """
    A class to build a PyBaMM pipeline for a given model and data, and run the resultant simulation.

    There are two contexts in which this class can be used:
    1. A pybamm model needs to be built once, and then run multiple times with different input parameters
    2. A pybamm model needs to be built multiple times with different parameter values,
        for the case where some of the parameters are geometric parameters which change the mesh

    To enable 2., you can pass a list of parameter names to the constructor, these parameters will be set
    before the model is built each time (using the `build` method).
    To enable 1, you can just pass an empty list. The model will be built once and subsequent calls
    to the `build` method will not change the model.
    """

    def __init__(
        self,
        model: pybamm.BaseModel,
        parameter_values: pybamm.ParameterValues = None,
        pybop_parameters: Parameters = None,
        solver: pybamm.BaseSolver = None,
        f_eval: list = None,
        var_pts: dict = None,
    ):
        """
        Arguments
        ---------
        model : pybamm.BaseModel
            The PyBaMM model to be used.
        parameter_values : pybamm.ParameterValues
            The parameters to be used in the model.
        solver : pybamm.BaseSolver
            The solver to be used. If None, the idaklu solver will be used.
        t_start : number
            The start time of the simulation.
        t_end : number
            The end time of the simulation.
        t_interp : np.ndarray
            The time points at which to interpolate the solution. If None, no interpolation will be done.
        rebuild_parameters : list[str]
            The parameters that will be used to rebuild the model. If None, the model will not be rebuilt.
        """
        super().__init__(
            model,
            parameter_values=parameter_values,
            pybop_parameters=pybop_parameters,
            solver=solver,
            var_pts=var_pts,
        )

        # Set-up model for EIS
        self._f_eval = f_eval
        self._model = self.set_up_for_eis(model.new_copy())
        self._parameter_values["Current function [A]"] = 0

        v_scale = getattr(model.variables["Voltage [V]"], "scale", 1)
        i_scale = getattr(model.variables["Current [A]"], "scale", 1)
        self.z_scale = self._parameter_values.evaluate(v_scale / i_scale)

    def set_up_for_eis(self, model):
        """
        Set up the model for electrochemical impedance spectroscopy (EIS) simulations.
        This method sets up the model for EIS simulations by adding the necessary
        algebraic equations and variables to the model.
        Originally developed by pybamm-eis: https://github.com/pybamm-team/pybamm-eis
        Parameters
        ----------
        model : pybamm.Model
            The PyBaMM model to be used for EIS simulations.
        """
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
        # Variable objects
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

    def initialise_eis_pipeline(self):
        """
        Initialise the Electrochemical Impedance Spectroscopy (EIS) simulation.
        This method sets up the mass matrix and solver, converts inputs to the appropriate format,
        extracts necessary attributes from the model, and prepares matrices for the simulation.
        """
        inputs = self._pybop_parameters.as_dict()
        M = self.built_model.mass_matrix.entries
        self._solver.set_up(self.built_model, inputs=inputs)

        # Convert inputs to casadi format if needed
        casadi_inputs = (
            casadi.vertcat(*inputs.values())
            if inputs is not None and self.built_model.convert_to_format == "casadi"
            else inputs or []
        )

        # Extract necessary attributes from the model
        y0 = self.built_model.concatenated_initial_conditions.evaluate(0, inputs=inputs)
        jac = self.built_model.jac_rhs_algebraic_eval(0, y0, casadi_inputs).sparse()

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
            frequency (np.ndarray | list like): The frequency at which to calculate the impedance.

        Returns
        -------
            The calculated impedance (complex np.ndarray).
        """
        # Compute the system matrix
        A = 1.0j * 2 * np.pi * frequency * self.M - self._jac

        # Solve the system
        x = spsolve(A, self.b)

        # Calculate the impedance
        return -x[-2] / x[-1]

    def solve(self, calculate_sensitivities: bool = False) -> np.ndarray:
        """
        Run the simulation using the built model and solver.

        Parameters
        ---------
        calculate_sensitivities : bool
            Whether to calculate sensitivities or not.

        Returns
        -------
        sol : dictionary
            A dictionary containing the impedance values with corresponding frequencies.
        """
        zs = [self.calculate_impedance(frequency) for frequency in self._f_eval]

        return np.asarray(zs) * self.z_scale
