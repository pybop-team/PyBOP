import numpy as np
import pybamm
import pytest

import pybop


class TestModels:
    """
    A class to test pybop created models.
    """

    pytestmark = pytest.mark.unit

    def test_weppner_huggins(self):
        model = pybop.lithium_ion.WeppnerHuggins()
        assert model is not None

    def test_grouped_spme_create_grouped_parameters(self):
        parameter_values = pybamm.ParameterValues("Chen2020")
        grouped_values = pybop.lithium_ion.GroupedSPMe.create_grouped_parameters(
            parameter_values
        )
        assert isinstance(grouped_values, pybamm.ParameterValues)

        model = pybop.lithium_ion.GroupedSPMe()
        variable_list = model.default_quick_plot_variables
        assert isinstance(variable_list, list)

    @pytest.fixture
    def half_cell_model(self):
        options = {"working electrode": "positive"}
        model = pybamm.lithium_ion.SPM(options=options)
        return model

    @pytest.fixture
    def parameter_values(self, half_cell_model):
        params = half_cell_model.default_parameter_values
        params.update(
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
        self.ground_truth = params["Positive electrode active material volume fraction"]
        return params

    @pytest.fixture
    def dataset(self, half_cell_model, parameter_values):
        sim = pybamm.Simulation(
            model=half_cell_model, parameter_values=parameter_values
        )
        sol = sim.solve(t_eval=[0, 10], t_interp=np.linspace(0, 10, 10))
        return pybop.Dataset(
            {
                "Time [s]": sol.t,
                "Current function [A]": sol["Current [A]"].data,
                "Voltage [V]": sol["Voltage [V]"].data,
            }
        )

    @pytest.fixture
    def fit_problem(self, half_cell_model, parameter_values, dataset):
        builder = pybop.builders.Pybamm()
        builder.add_parameter(
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                initial_value=0.75,
            )
        )
        builder.set_dataset(dataset)
        builder.set_simulation(half_cell_model, parameter_values)
        builder.add_cost(
            pybop.costs.pybamm.MeanAbsoluteError("Voltage [V]", "Voltage [V]")
        )
        return builder.build()

    def test_fitting_cost(self, fit_problem):
        # Get initial cost
        initial_cost = fit_problem.run()

        # Get ground truth cost
        fit_problem.set_params(np.asarray([self.ground_truth]))
        ground_truth_cost = fit_problem.run()

        assert initial_cost > ground_truth_cost

    @pytest.fixture
    def design_problem(self, half_cell_model, parameter_values, dataset):
        builder = pybop.builders.Pybamm()
        builder.add_parameter(
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                initial_value=0.75,
            )
        )
        builder.set_dataset(dataset)
        builder.set_simulation(half_cell_model, parameter_values)
        builder.add_cost(pybop.costs.pybamm.GravimetricEnergyDensity())
        return builder.build()

    def test_design_cost(self, design_problem):
        # Get initial cost
        initial_cost = design_problem.run()

        # Get ground truth cost
        design_problem.set_params(np.asarray([self.ground_truth]))
        ground_truth_cost = design_problem.run()

        assert initial_cost < ground_truth_cost  # Negative cost
