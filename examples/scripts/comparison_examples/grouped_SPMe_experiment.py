import pybop
from pybop.models.lithium_ion.basic_SPMe import convert_physical_to_grouped_parameters


# Define parameter set
parameter_set = pybop.ParameterSet.pybamm("Chen2020")
parameter_set["Electrolyte diffusivity [m2.s-1]"] = 1.769e-10
parameter_set["Electrolyte conductivity [S.m-1]"] = 1e16
parameter_set["Negative electrode conductivity [S.m-1]"] = 1e16
parameter_set["Positive electrode conductivity [S.m-1]"] = 1e16
grouped_parameters = convert_physical_to_grouped_parameters(parameter_set)
grouped_parameters["Series resistance [Ohm]"] = 0.01

## Create model
var_pts = {"x_n": 20, "x_s": 20, "x_p": 20, "r_n": 20, "r_p": 20}
model_options = {"surface form": "differential", "contact resistance": "true"}
model = pybop.lithium_ion.GroupedSPMe(
    parameter_set=grouped_parameters, var_pts=var_pts, options=model_options
)

## Test model in the time domain
model.set_initial_state({"Initial SoC": 0.9})
experiment = pybop.Experiment(
    [
        "Rest for 10 minutes (10 seconds period)",
        # "Discharge at 1C until 2.5 V (10 seconds period)",
        "Rest for 20 minutes (10 seconds period)",
    ],
)
simulation = model.predict(experiment=experiment)
dataset = pybop.Dataset(
    {
        "Time [s]": simulation["Time [s]"].data,
        "Current function [A]": simulation["Current [A]"].data,
        "Voltage [V]": simulation["Voltage [V]"].data,
    }
)
pybop.plot.dataset(dataset, signal=["Voltage [V]", "Current function [A]"])
