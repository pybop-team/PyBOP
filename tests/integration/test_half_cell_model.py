import numpy as np
import pybamm
import pytest
from pybamm import Parameter

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
        options = {"working electrode": "positive"}
        parameter_set = pybop.lithium_ion.SPM(options=options).default_parameter_values
        parameter_set.update(
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
            },
            check_already_exists=False,
        )
        x = self.ground_truth
        parameter_set.update(
            {
                "Positive electrode active material volume fraction": x[0],
            }
        )
        return pybop.lithium_ion.SPM(parameter_set=parameter_set, options=options)

    @pytest.fixture
    def parameters(self):
        return pybop.Parameters(
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                prior=pybop.Uniform(0.4, 0.75),
                # no bounds
            ),
        )

    @pytest.fixture(params=[0.4])
    def init_soc(self, request):
        return request.param

    def noisy(self, data, sigma):
        return data + np.random.normal(0, sigma, len(data))

    @pytest.fixture
    def fitting_cost(self, model, parameters, init_soc):
        # Form dataset
        solution = self.get_data(model, init_soc)
        dataset = pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": self.noisy(solution["Voltage [V]"].data, self.sigma0),
            }
        )

        # Define the cost to optimise
        problem = pybop.FittingProblem(model, parameters, dataset)
        return pybop.SumSquaredError(problem)

    def test_fitting_costs(self, fitting_cost):
        x0 = fitting_cost.parameters.get_initial_values()
        optim = pybop.CuckooSearch(
            cost=fitting_cost,
            sigma0=0.03,
            max_iterations=250,
            max_unchanged_iterations=35,
        )

        initial_cost = optim.cost(optim.parameters.get_initial_values())
        results = optim.run()

        # Assertions
        if not np.allclose(x0, self.ground_truth, atol=1e-5):
            if results.minimising:
                assert initial_cost > results.final_cost
            else:
                assert initial_cost < results.final_cost
        np.testing.assert_allclose(results.x, self.ground_truth, atol=1.5e-2)

    @pytest.fixture
    def design_cost(self, model):
        initial_state = {"Initial SoC": 1.0}
        parameters = pybop.Parameters(
            pybop.Parameter(
                "Positive electrode thickness [m]",
                prior=pybop.Gaussian(5e-05, 5e-06),
                bounds=[2e-06, 10e-05],
            ),
        )
        experiment = pybamm.Experiment(
            ["Discharge at 1C until 3.5 V (5 seconds period)"],
        )

        problem = pybop.DesignProblem(
            model,
            parameters,
            experiment=experiment,
            initial_state=initial_state,
        )
        return pybop.GravimetricEnergyDensity(problem)

    def test_design_costs(self, design_cost):
        optim = pybop.CuckooSearch(
            design_cost,
            max_iterations=15,
            allow_infeasible_solutions=False,
        )
        initial_values = optim.parameters.get_initial_values()
        initial_cost = optim.cost(initial_values)
        results = optim.run()

        # Assertions
        assert initial_cost < results.final_cost
        for i, _ in enumerate(results.x):
            assert results.x[i] < initial_values[i]

    def get_data(self, model, init_soc):
        initial_state = {"Initial SoC": init_soc}
        experiment = pybamm.Experiment(
            [
                (
                    "Discharge at 0.5C for 3 minutes (4 second period)",
                    "Charge at 0.5C for 3 minutes (4 second period)",
                ),
            ]
        )
        sim = model.predict(initial_state=initial_state, experiment=experiment)
        return sim
