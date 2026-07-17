import pickle
from pathlib import Path

from pybop.applications.utils import get_cells

"""
Our data is stored in this folder structure:
examples/data
    - {Cell type}
        - {Cell format}
            - {Cell label}
                - {Procedure}.parquet
                - metadata.json

For each full-cell, locate the corresponding half-cell data and create a parameters
dictionary for charge and discharge.
"""

for cell in get_cells():
    cell_label = cell.info["Cell label"]
    cell_type = cell.info["Cell type"]
    cell_path = Path(__file__).parent / "results" / cell_type / cell_label
    cell_path.mkdir(parents=True, exist_ok=True)

    # Create shared parameters
    param = {
        "chemistry": "lithium_ion",
        "Nominal cell capacity [A.h]": cell.info["Nominal cell capacity [A.h]"],
        "Ambient temperature [K]": 298.15,
        "Initial temperature [K]": 298.15,
        "Number of electrodes connected in parallel to make a cell": 1.0,
        "Number of cells connected in series to make a battery": 1.0,
    }

    # Add cell parameters
    if cell_type == "LithiumWerks M1B":
        three_layer_thickness = 59 + 19.5 + 35.7
        param.update(
            {
                "Lower voltage cut-off [V]": 2.0,
                "Upper voltage cut-off [V]": 3.6,
                "Positive electrode relative thickness": 59 / three_layer_thickness,
                "Negative electrode relative thickness": 35.7 / three_layer_thickness,
            }
        )
    elif cell_type == "Molicel P45B":
        three_layer_thickness = 39 + 12 + 50.8
        param.update(
            {
                "Lower voltage cut-off [V]": 2.5,
                "Upper voltage cut-off [V]": 4.2,
                "Positive electrode relative thickness": 39 / three_layer_thickness,
                "Negative electrode relative thickness": 50.8 / three_layer_thickness,
            }
        )
    elif cell_type == "LG M50 Synthetic":
        three_layer_thickness = 75.6 + 12 + 85.2
        param.update(
            {
                "Lower voltage cut-off [V]": 2.5,
                "Upper voltage cut-off [V]": 4.2,
                "Positive electrode relative thickness": 75.6 / three_layer_thickness,
                "Negative electrode relative thickness": 85.2 / three_layer_thickness,
            }
        )
    else:
        raise ValueError(f"Unrecognised cell type: {cell_type}")

    with open(cell_path / "params_charge.pickle", "wb") as file:
        pickle.dump(param, file)
    with open(cell_path / "params_discharge.pickle", "wb") as file:
        pickle.dump(param, file)
