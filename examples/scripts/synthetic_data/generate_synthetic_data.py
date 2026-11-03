"""
Generate synthetic battery data using PyBaMM from JSON spec files.

Outputs are written to:
  - examples/data/<Cell type>/<Cell format>/<Cell label>/
"""

import json

import numpy as np
import pybamm

import pybop

# Define paths relative to the pybop directory
procedure_root = pybop.script_path / "../examples/scripts/synthetic_data/procedures"
archive_root = pybop.script_path / "../examples/data"

model_class = pybamm.lithium_ion.DFN
full_cell_parameters = pybamm.ParameterValues("Chen2020")
cell_type = "LG M50 Synthetic"
cell_label = "C02"
Qnom_neg, Vmin_neg, Vmax_neg = 6.0, 0.05, 1.5
Qnom_pos, Vmin_pos, Vmax_pos = 7.0, 3.5, 4.4

""" Determine the capacity of the full-cell. """
cell_info = {
    "Cell type": cell_type,
    "Cell format": "full cell",
    "Cell label": cell_label,
}
model_options = {"thermal": "lumped", "contact resistance": "true"}
full_cell_model = model_class(model_options)
cell = pybop.pybamm.simulate_procedure(
    info=cell_info,
    model=full_cell_model,
    parameter_values=full_cell_parameters,
    spec_path=procedure_root / "Capacity determination.json",
)
throughput = (
    cell.procedure["Capacity determination"]
    .experiment("Capacity determination")
    .get("Capacity [Ah]")
)
Q_meas = np.round((throughput[0] - throughput[-1]), decimals=3)

""" Generate time-domain data for the full-cell. """
full_cell_parameters["Nominal cell capacity [A.h]"] = Q_meas
procedures = [
    "pOCV.json",
    "Temperature experiment.json",
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

""" Generate EIS data for the full-cell. """
cell_info = {
    "Cell type": cell_type,
    "Cell format": "full cell",
    "Cell label": "EIS",
}
model_options.update({"surface form": "differential"})
full_cell_eis_model = model_class(model_options)
cell = pybop.pybamm.simulate_procedure(
    info=cell_info,
    model=full_cell_eis_model,
    parameter_values=full_cell_parameters,
    spec_path=procedure_root / "EIS charge.json",
)
pybop.pybamm.archive_data(cell=cell, archive_root=archive_root)

""" Generate time-domain data for the negative electrode half-cell. """
cell_info = {
    "Cell type": cell_type,
    "Cell format": "negative half cell",
    "Cell label": "neg_01",
}
model_options = {
    "working electrode": "positive"
}  # PyBaMM uses "positive" for all half-cells
negative_electrode_model = model_class(model_options)
negative_electrode_parameters = pybop.pybamm.convert_to_half_cell_parameters(
    full_cell_parameters, "negative"
)
negative_electrode_parameters.update(
    {
        "Nominal cell capacity [A.h]": Qnom_neg,
        "Lower voltage cut-off [V]": Vmin_neg,
        "Upper voltage cut-off [V]": Vmax_neg,
    }
)
cell = pybop.pybamm.simulate_procedure(
    info=cell_info,
    model=negative_electrode_model,
    parameter_values=negative_electrode_parameters,
    spec_path=procedure_root / "pOCP negative.json",
)
pybop.pybamm.archive_data(cell=cell, archive_root=archive_root)

""" Generate time-domain data for the positive electrode half-cell. """
cell_info = {
    "Cell type": cell_type,
    "Cell format": "positive half cell",
    "Cell label": "pos_01",
}
model_options = {"working electrode": "positive"}
positive_electrode_model = model_class(model_options)
positive_electrode_parameters = pybop.pybamm.convert_to_half_cell_parameters(
    full_cell_parameters, "positive"
)
positive_electrode_parameters.update(
    {
        "Nominal cell capacity [A.h]": Qnom_pos,
        "Lower voltage cut-off [V]": Vmin_pos,
        "Upper voltage cut-off [V]": Vmax_pos,
    }
)
cell = pybop.pybamm.simulate_procedure(
    info=cell_info,
    model=positive_electrode_model,
    parameter_values=positive_electrode_parameters,
    spec_path=procedure_root / "pOCP positive.json",
)
pybop.pybamm.archive_data(cell=cell, archive_root=archive_root)

# Associate all the synthetic data with the full-cell
archive = (
    archive_root / cell_info["Cell type"] / "Full cell" / cell_label / "metadata.json"
)
associated_data = {
    "Negative electrode": "../../Half cell negative/neg_01",
    "Positive electrode": "../../Half cell positive/pos_01",
    "Full-cell EIS": "../EIS",
}

with open(archive) as file:
    metadata = json.load(file)
metadata["info"]["Associated"] = associated_data

json_clean = json.dumps(metadata, indent=4)
with open(archive, "w") as file:
    file.write(json_clean + "\n")
