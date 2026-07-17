"""
Generate synthetic battery data using PyBaMM from JSON spec files.

Outputs are written to:
  - examples/data/<Cell type>/<Cell format>/<Cell label>/
"""

from pathlib import Path

import pybamm

import pybop

# Define the script's directory to resolve relative paths
SCRIPT_DIR = Path(__file__).parent
procedure_root = SCRIPT_DIR / "procedures"
archive_root = SCRIPT_DIR.parent.parent / "data"

model_class = pybamm.lithium_ion.DFN
full_cell_parameters = pybamm.ParameterValues("Chen2020")

""" Generate time-domain data for the LG M50. """
cell_info = {
    "Cell type": "LG M50 Synthetic",
    "Cell format": "full cell",
    "Cell label": "C02",
}
model_options = None
full_cell_model = model_class(model_options)
full_cell_parameters["Nominal cell capacity [A.h]"] = 5.0
procedures = [
    "Capacity determination.json",
    "pOCV.json",
    "GITT.json",
    "Validation cycling.json",
    "Validation different SOC.json",
    "Validation pulses.json",
]
cell = pybop.pybamm.simulate_procedure(
    info=cell_info,
    model=full_cell_model,
    parameter_values=full_cell_parameters,
    spec_path=[procedure_root / p for p in procedures],
)
pybop.pybamm.archive_data(cell=cell, archive_root=archive_root)

""" Generate EIS data for the LG M50. """
cell_info = {
    "Cell type": "LG M50 Synthetic",
    "Cell format": "full cell",
    "Cell label": "EIS",
}
model_options = {"surface form": "differential"}
full_cell_eis_model = model_class(model_options)
cell = pybop.pybamm.simulate_procedure(
    info=cell_info,
    model=full_cell_eis_model,
    parameter_values=full_cell_parameters,
    spec_path=procedure_root / "EIS charge.json",
)
pybop.pybamm.archive_data(cell=cell, archive_root=archive_root)

""" Generate time-domain data for the negative electrode of the LG M50. """
cell_info = {
    "Cell type": "LG M50 Synthetic",
    "Cell format": "negative half cell",
    "Cell label": "neg_01",
}
model_options = {
    "working electrode": "positive"
}  # PyBaMM uses "positive" for all half-cells
negative_electrode_model = model_class({"working electrode": "positive"})
negative_electrode_parameters = pybop.pybamm.convert_to_half_cell_parameters(
    full_cell_parameters, "negative"
)
negative_electrode_parameters["Nominal cell capacity [A.h]"] = 3.75
cell = pybop.pybamm.simulate_procedure(
    info=cell_info,
    model=negative_electrode_model,
    parameter_values=negative_electrode_parameters,
    spec_path=procedure_root / "pOCP negative.json",
)
pybop.pybamm.archive_data(cell=cell, archive_root=archive_root)

""" Generate time-domain data for the positive electrode of the LG M50. """
cell_info = {
    "Cell type": "LG M50 Synthetic",
    "Cell format": "positive half cell",
    "Cell label": "pos_01",
}
model_options = {"working electrode": "positive"}
positive_electrode_model = model_class({"working electrode": "positive"})
positive_electrode_parameters = pybop.pybamm.convert_to_half_cell_parameters(
    full_cell_parameters, "positive"
)
positive_electrode_parameters["Nominal cell capacity [A.h]"] = 6.5
cell = pybop.pybamm.simulate_procedure(
    info=cell_info,
    model=positive_electrode_model,
    parameter_values=positive_electrode_parameters,
    spec_path=procedure_root / "pOCP positive.json",
)
pybop.pybamm.archive_data(cell=cell, archive_root=archive_root)
