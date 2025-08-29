import numpy as np
import pybamm
import pytest

import pybop

# Test configuration constants
TIME_POINTS = 20
TIME_MAX = 100
FREQ_POINTS = 30
TEST_PARAM_VALUES = np.asarray([550, 550])
RELATIVE_TOLERANCE = 1e-5
ABSOLUTE_TOLERANCE = 1e-5

# Parameter configurations
CHARGE_TRANSFER_PARAMS = [
    ("Positive electrode charge transfer time scale [s]", 500),
    ("Negative electrode charge transfer time scale [s]", 500),
]


class TestHalfCell:
    """
    A class to test a half-cell model.
    """

    pytestmark = pytest.mark.unit

    @pytest.fixture
    def model_config(self, request):
        """Shared model configuration to avoid repeated initialisation."""
        options = {"working electrode": "positive"}
        model = pybamm.lithium_ion.SPMe(options=options)
        parameter_values = pybamm.ParameterValues("Xu2019")
        parameter_values.update(
            {
                "Electrolyte density [kg.m-3]": pybamm.Parameter(
                    "Positive electrode density [kg.m-3]"
                ),
                "Negative current collector density [kg.m-3]": 0.0,
                "Negative current collector thickness [m]": 0.0,
                "Negative electrode active material density [kg.m-3]": 0.0,
                "Negative electrode active material volume fraction": 0.0,
                "Negative electrode porosity": 0.0,
                "Negative electrode carbon-binder density [kg.m-3]": 0.0,
                "Positive current collector density [kg.m-3]": 0.0,
                "Positive current collector thickness [m]": 0.0,
                "Positive electrode active material density [kg.m-3]": pybamm.Parameter(
                    "Positive electrode density [kg.m-3]"
                ),
                "Positive electrode carbon-binder density [kg.m-3]": pybamm.Parameter(
                    "Positive electrode density [kg.m-3]"
                ),
                "Positive electrode density [kg.m-3]": 3262.0,
                "Separator density [kg.m-3]": 0.0,
            },
            check_already_exists=False,
        )
        self.ground_truth = [
            parameter_values["Positive electrode active material volume fraction"]
        ]

        return {"model": model, "parameter_values": parameter_values}

    @pytest.fixture
    def parameters(self):
        """Create parameter objects for reuse."""
        return [
            pybop.Parameter(
                "Positive electrode active material volume fraction", initial_value=0.75
            )
        ]

    def test_fitting(self, model_config, parameters, r_solver):
        sim = pybamm.Simulation(
            model_config["model"], parameter_values=model_config["parameter_values"]
        )
        sol = sim.solve(t_eval=[0, 10], t_interp=np.linspace(0, 10, 10))
        dataset = pybop.Dataset(
            {
                "Time [s]": sol.t,
                "Current function [A]": sol["Current [A]"].data,
                "Voltage [V]": sol["Voltage [V]"].data,
            }
        )

        builder = pybop.builders.Pybamm()
        for parameter in parameters:
            builder.add_parameter(parameter)
        builder.set_dataset(dataset)
        builder.set_simulation(
            model_config["model"],
            parameter_values=model_config["parameter_values"],
            solver=r_solver,
        )
        builder.add_cost(
            pybop.costs.pybamm.MeanAbsoluteError("Voltage [V]", "Voltage [V]")
        )
        fitting_problem = builder.build()

        # Get initial cost
        initial_cost = fitting_problem.run()

        # Get ground truth cost
        fitting_problem.set_params(np.asarray(self.ground_truth))
        ground_truth_cost = fitting_problem.run()

        assert initial_cost > ground_truth_cost

    def test_design(self, model_config, parameters, r_solver):
        model = model_config["model"]
        parameter_values = model_config["parameter_values"]
        pybop.builders.cell_mass(parameter_values)

        experiment = pybamm.Experiment(
            [
                "Discharge at 1C until 2.5 V (10 seconds period)",
                "Rest for 10 minutes (10 seconds period)",
            ],
        )

        builder = pybop.builders.Pybamm()
        for parameter in parameters:
            builder.add_parameter(parameter)
        builder.set_simulation(
            model,
            parameter_values=parameter_values,
            experiment=experiment,
            solver=r_solver,
        )
        builder.add_cost(pybop.costs.pybamm.GravimetricEnergyDensity())
        design_problem = builder.build()

        # Get initial cost
        initial_cost = design_problem.run()

        # Get ground truth cost
        design_problem.set_params(np.asarray(self.ground_truth))
        ground_truth_cost = design_problem.run()

        assert initial_cost < ground_truth_cost  # negative cost
