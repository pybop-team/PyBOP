from typing import Optional

import casadi
import numpy as np
import pybamm
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

from pybop import SymbolReplacer
from pybop.parameters.parameter import Inputs


def set_up_for_eis(model):
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
        model.param, None, model.options, control="algebraic"
    ).get_fundamental_variables()

    # Define the variables to replace
    symbol_replacement_map = {}
    for name, variable in external_circuit_variables.items():
        if name in model.variables.keys():
            symbol_replacement_map[model.variables[name]] = variable

    # Don't replace initial conditions, as these should not contain
    # Variable objects
    replacer = SymbolReplacer(symbol_replacement_map, process_initial_conditions=False)
    replacer.process_model(model, inplace=True)

    # Add an algebraic equation for the current density variable
    # External circuit submodels are always equations on the current
    I_cell = model.variables["Current variable [A]"]
    I = model.variables["Current [A]"]
    I_applied = pybamm.FunctionParameter("Current function [A]", {"Time [s]": pybamm.t})
    model.algebraic[I_cell] = I - I_applied
    model.initial_conditions[I_cell] = 0


def initialise_eis_simulation(model, inputs: Optional[Inputs] = None):
    """
    Initialise the Electrochemical Impedance Spectroscopy (EIS) simulation.

    This method sets up the mass matrix and solver, converts inputs to the appropriate format,
    extracts necessary attributes from the model, and prepares matrices for the simulation.

    Parameters
    ----------
    inputs : dict (optional)
        The input parameters for the simulation.
    """
    # Setup mass matrix, solver
    M = model.built_model.mass_matrix.entries
    model.solver.set_up(model.built_model, inputs=inputs)

    # Convert inputs to casadi format if needed
    casadi_inputs = (
        casadi.vertcat(*inputs.values())
        if inputs is not None and model.built_model.convert_to_format == "casadi"
        else inputs or []
    )

    # Extract necessary attributes from the model
    y0 = model.built_model.concatenated_initial_conditions.evaluate(0, inputs=inputs)
    J = model.built_model.jac_rhs_algebraic_eval(0, y0, casadi_inputs).sparse()

    # Convert to Compressed Sparse Column format
    M = csc_matrix(M)
    J = csc_matrix(J)

    # Add forcing to the RHS on the current density
    b = np.zeros(y0.shape)
    b[-1] = -1

    return M, J, b


def calculate_impedance(M, J, b, frequency):
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
    A = 1.0j * 2 * np.pi * frequency * M - J

    # Solve the system
    x = spsolve(A, b)

    # Calculate the impedance
    return -x[-2] / x[-1]
