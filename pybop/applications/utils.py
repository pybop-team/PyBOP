from collections.abc import Callable

import polars as pl

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
