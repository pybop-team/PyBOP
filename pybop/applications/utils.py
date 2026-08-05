from collections.abc import Callable

import numpy as np
import polars as pl
from scipy.optimize import minimize

from pybop import Interpolant, script_path


class OpenCircuitVoltage:
    def __init__(
        self,
        positive_ocp_function: Callable,
        sto_p_0: float,
        sto_p_100: float,
        negative_ocp_function: Callable,
        sto_n_0: float,
        sto_n_100: float,
    ):
        self.positive_ocp_function = positive_ocp_function
        self.sto_p_0 = sto_p_0
        self.sto_p_100 = sto_p_100
        self.negative_ocp_function = negative_ocp_function
        self.sto_n_0 = sto_n_0
        self.sto_n_100 = sto_n_100

    def __call__(self, soc):
        return self.positive_ocp_function(
            self.sto_p_0 + (self.sto_p_100 - self.sto_p_0) * soc
        ) - self.negative_ocp_function(
            self.sto_n_0 + (self.sto_n_100 - self.sto_n_0) * soc
        )


def get_cells(match: str = "C0"):
    import pyprobe

    # Define paths relative to the pybop directory
    archive_root = script_path / "../examples/data"

    if not archive_root.is_dir():
        raise FileNotFoundError(f"No such directory: {archive_root}")

    # Find all archives
    cell_archives = sorted(list(archive_root.rglob("metadata.json")))

    cells = []
    for archive in cell_archives:
        cell = pyprobe.load_archive(str(archive.parent))
        if match in cell.info["Cell label"]:
            cell.info["Archive"] = archive.parent
            cells.append(cell)

    return cells


def filter_with_preceding_row(
    procedure,
    experiment: str | None = None,
    cycle: int | None = None,
    phase: str | None = None,
    step: int | None = None,
):
    """
    Obtain the PyProBE result including the preceding row of the data.
    """
    data = procedure
    if experiment is not None:
        data = data.experiment(experiment)
    if cycle is not None:
        data = data.cycle(cycle)
    if phase is not None:
        if phase == "discharge":
            data = data.discharge()
        elif phase == "charge":
            data = data.charge()
        elif phase == "rest":
            data = data.rest()
        else:
            raise ValueError("Unrecognised phase")
    if step is not None:
        data = data.step(step)

    # Add the preceding row
    start_time = data.lf.select("Time [s]").first().collect().item()
    preceding_row = procedure.lf.filter(pl.col("Time [s]") < start_time).last()
    data.lf = pl.concat((preceding_row, data.lf), how="diagonal_relaxed")

    return data


def make_voltage_monotonic(data):
    """
    Filter a pseudo-OCV measurement so that it is monotonic.
    """
    initial_voltage = data.lf.select("Voltage [V]").first().collect().item()
    final_voltage = data.lf.select("Voltage [V]").last().collect().item()
    if initial_voltage > final_voltage:
        # Get the cumulative maximum voltage, applied from the end of discharge
        filtered_voltage = (
            pl.col("Voltage [V]").cum_max(reverse=True).alias("Filtered voltage [V]")
        )
        # Replace the voltage column with the filtered voltage
        data.lf = (
            data.lf.with_columns(filtered_voltage=filtered_voltage)
            .collect()
            .drop("Voltage [V]")
            .rename({"filtered_voltage": "Voltage [V]"})
        )
    elif initial_voltage < final_voltage:
        # Get the cumulative minimum voltage, applied from the end of charge
        filtered_voltage = (
            pl.col("Voltage [V]").cum_min(reverse=True).alias("Filtered voltage [V]")
        )
        # Replace the voltage column with the filtered voltage
        data.lf = (
            data.lf.with_columns(filtered_voltage=filtered_voltage)
            .collect()
            .drop("Voltage [V]")
            .rename({"filtered_voltage": "Voltage [V]"})
        )
    else:
        print("Initial and final voltages are the same.")
    return data


def get_ocp_functions(parameter_values, OCP_type: str = None):
    """Returns callable OCP functions from a lithium-ion parameter set."""
    OCP_type = "OCP" if OCP_type is None else OCP_type

    # Ensure that a pybop.Interpolant is used instead of a pybamm.Interpolant
    positive_ocp_function = Interpolant(
        parameter_values[f"Positive electrode {OCP_type} [V]"].x,
        parameter_values[f"Positive electrode {OCP_type} [V]"].y,
        name=f"Positive {OCP_type}",
    )
    negative_ocp_function = Interpolant(
        parameter_values[f"Negative electrode {OCP_type} [V]"].x,
        parameter_values[f"Negative electrode {OCP_type} [V]"].y,
        name=f"Negative {OCP_type}",
    )
    return positive_ocp_function, negative_ocp_function


def get_ocv_function(parameter_values, OCP_type: str = None):
    """Returns a callable OCV function from a lithium-ion parameter set."""
    OCP_type = "OCP" if OCP_type is None else OCP_type

    x_0 = parameter_values["Minimum negative stoichiometry"]
    x_100 = parameter_values["Maximum negative stoichiometry"]
    y_100 = parameter_values["Minimum positive stoichiometry"]
    y_0 = parameter_values["Maximum positive stoichiometry"]

    positive_ocp_function, negative_ocp_function = get_ocp_functions(
        parameter_values, OCP_type
    )
    return lambda soc: (
        positive_ocp_function(y_0 + (y_100 - y_0) * soc)
        - negative_ocp_function(x_0 + (x_100 - x_0) * soc)
    )


def shift_ocv_to(voltage_points, parameter_values, naive_soc, OCP_type: str = None):
    """Returns a new lithium-ion parameter set with shifted OCP functions and SOC points."""
    OCP_type = "OCP" if OCP_type is None else OCP_type

    param = parameter_values.copy()
    ocv_function = get_ocv_function(param, OCP_type)

    def mean_absolute_error(values):
        shift, stretch, offset = values
        return np.mean(
            np.abs(voltage_points - ocv_function(shift + stretch * naive_soc) - offset)
        )

    x0 = [0, 1, 0]
    scipy_result = minimize(mean_absolute_error, x0=x0)
    SOC_vec = scipy_result.x[0] + scipy_result.x[1] * naive_soc

    # Shift each of the OCP functions by half the offset to remove series resistance
    offset = scipy_result.x[2]
    param["Positive electrode OCP [V]"] = Interpolant(
        param[f"Positive electrode {OCP_type} [V]"].x,
        param[f"Positive electrode {OCP_type} [V]"].y + offset / 2,
        name="Positive OCP",
    )
    param["Negative electrode OCP [V]"] = Interpolant(
        param[f"Negative electrode {OCP_type} [V]"].x,
        param[f"Negative electrode {OCP_type} [V]"].y - offset / 2,
        name="Negative OCP",
    )

    return param, SOC_vec
