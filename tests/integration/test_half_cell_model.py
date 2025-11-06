import numpy as np
import pybamm
import pytest
from pybamm import Parameter
from scipy import stats

import pybop


class TestHalfCellModel:
    """
    A class to test optimisation of a PyBaMM half-cell model.
    """

    pytestmark = pytest.mark.integration

    @pytest.fixture(autouse=True)
    def setup(self):
        self.sigma0 = 0.002
        self.ground_truth = np.clip(
            np.asarray([0.5]) + np.random.normal(loc=0.0, scale=0.05, size=1),
            a_min=0.4,
            a_max=0.75,
        )

    @pytest.fixture
    def model(self):
        model = pybamm.lithium_ion.SPM(options={"working electrode": "positive"})
        pybop.pybamm.add_variable_to_model(
            model, "Gravimetric energy density [Wh.kg-1]"
        )
        return model

    @pytest.fixture
    def parameter_values(self, model):
        parameter_values = model.default_parameter_values
        parameter_values.update(
            {
                "Electrolyte density [kg.m-3]": Parameter(
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
                "Positive electrode active material density [kg.m-3]": Parameter(
                    "Positive electrode density [kg.m-3]"
                ),
                "Positive electrode carbon-binder density [kg.m-3]": Parameter(
                    "Positive electrode density [kg.m-3]"
                ),
                "Positive electrode density [kg.m-3]": 3262.0,
                "Separator density [kg.m-3]": 0.0,
                "Cell mass [kg]": pybop.pybamm.cell_mass(),
            },
            check_already_exists=False,
        )
        x = self.ground_truth
        parameter_values.update(
            {"Positive electrode active material volume fraction": x[0]}
        )
        return parameter_values

    @pytest.fixture
    def parameters(self):
        return {
            "Positive electrode active material volume fraction": pybop.ParameterDistribution(
                stats.uniform(0.4, 0.75 - 0.4),
                # no bounds
            ),
        }

    def noisy(self, data, sigma):
        return data + np.random.normal(0, sigma, len(data))

    @pytest.fixture
    def fitting_problem(self, model, parameter_values, parameters):
        parameter_values.set_initial_state(0.4, options=model.options)
        dataset = self.get_data(model, parameter_values)

        # Define the cost to optimise
        parameter_values.update(parameters)
        simulator = pybop.pybamm.Simulator(
            model, parameter_values=parameter_values, protocol=dataset
        )
        cost = pybop.SumSquaredError(dataset)
        return pybop.Problem(simulator, cost)

    def test_fitting_costs(self, fitting_problem):
        x0 = fitting_problem.parameters.get_initial_values()
        options = pybop.PintsOptions(
            sigma=0.03,
            max_iterations=250,
            max_unchanged_iterations=35,
        )
        optim = pybop.CuckooSearch(fitting_problem, options=options)
        results = optim.run()

        # Assertions
        initial_cost = optim.problem(x0)
        if not np.allclose(x0, self.ground_truth, atol=1e-5):
            if results.minimising:
                assert initial_cost > results.best_cost
            else:
                assert initial_cost < results.best_cost
        np.testing.assert_allclose(results.x, self.ground_truth, atol=1.5e-2)

    @pytest.fixture
    def design_problem(self, model, parameter_values):
        pybop.pybamm.set_formation_concentrations(parameter_values)
        initial_state = {"Initial SoC": 1.0}
        parameter_values.update(
            {
                "Positive electrode thickness [m]": pybop.TruncatedGaussian(
                    loc=5e-05,
                    scale=5e-06,
                    bounds=[2e-06, 10e-05],
                ),
            }
        )
        experiment = pybamm.Experiment(
            ["Discharge at 1C until 3.5 V (5 seconds period)"]
        )
        simulator = pybop.pybamm.Simulator(
            model,
            parameter_values=parameter_values,
            protocol=experiment,
            initial_state=initial_state,
        )
        cost = pybop.DesignCost(target="Gravimetric energy density [Wh.kg-1]")
        return pybop.Problem(simulator, cost)

    def test_design_costs(self, design_problem):
        options = pybop.PintsOptions(max_iterations=15)
        optim = pybop.CuckooSearch(design_problem, options=options)
        initial_values = optim.problem.parameters.get_initial_values()
        initial_cost = optim.problem(initial_values)
        results = optim.run()

        # Assertions
        assert initial_cost < results.best_cost

    def get_data(self, model, parameter_values):
        experiment = pybamm.Experiment(
            [
                "Discharge at 0.5C for 3 minutes (4 second period)",
                "Charge at 0.5C for 3 minutes (4 second period)",
            ]
        )
        solution = pybamm.Simulation(
            model, parameter_values=parameter_values, experiment=experiment
        ).solve()
        return pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": self.noisy(solution["Voltage [V]"].data, self.sigma0),
            }
        )
