import numpy as np
import pybamm
import pytest
from pybamm import Parameter

import pybop


class TestWeightedCost:
    """
    A class to test the weighted cost function.
    """

    pytestmark = pytest.mark.integration

    @pytest.fixture(autouse=True)
    def setup(self):
        self.sigma0 = 0.002
        self.ground_truth = np.clip(
            np.asarray([0.55, 0.55]) + np.random.normal(loc=0.0, scale=0.05, size=2),
            a_min=0.4,
            a_max=0.75,
        )

    @pytest.fixture
    def model(self):
        model = pybamm.lithium_ion.SPM()
        pybop.pybamm.add_variable_to_model(
            model, "Gravimetric energy density [Wh.kg-1]"
        )
        pybop.pybamm.add_variable_to_model(model, "Volumetric energy density [Wh.m-3]")
        return model

    @pytest.fixture
    def parameter_values(self):
        parameter_values = pybamm.ParameterValues("Chen2020")
        parameter_values.update(
            {
                "Electrolyte density [kg.m-3]": Parameter("Separator density [kg.m-3]"),
                "Negative electrode active material density [kg.m-3]": Parameter(
                    "Negative electrode density [kg.m-3]"
                ),
                "Negative electrode carbon-binder density [kg.m-3]": Parameter(
                    "Negative electrode density [kg.m-3]"
                ),
                "Positive electrode active material density [kg.m-3]": Parameter(
                    "Positive electrode density [kg.m-3]"
                ),
                "Positive electrode carbon-binder density [kg.m-3]": Parameter(
                    "Positive electrode density [kg.m-3]"
                ),
                "Cell mass [kg]": pybop.pybamm.cell_mass(),
                "Cell volume [m3]": pybop.pybamm.cell_volume(),
            },
            check_already_exists=False,
        )
        x = self.ground_truth
        parameter_values.update(
            {
                "Negative electrode active material volume fraction": x[0],
                "Positive electrode active material volume fraction": x[1],
            }
        )
        return parameter_values

    @pytest.fixture
    def parameters(self):
        return pybop.Parameters(
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Uniform(0.4, 0.75),
                bounds=[0.375, 0.75],
            ),
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                prior=pybop.Uniform(0.4, 0.75),
                # no bounds
            ),
        )

    @pytest.fixture(
        params=[
            (
                pybop.GaussianLogLikelihoodKnownSigma,
                pybop.RootMeanSquaredError,
                pybop.SumSquaredError,
            )
        ]
    )
    def cost_class(self, request):
        return request.param

    def noisy(self, data, sigma):
        return data + np.random.normal(0, sigma, len(data))

    @pytest.fixture
    def weighted_fitting_problem(self, model, parameter_values, parameters, cost_class):
        parameter_values.set_initial_state(0.4)
        dataset = self.get_data(model, parameter_values)

        # Define the problem
        simulator = pybop.pybamm.Simulator(
            model,
            parameter_values=parameter_values,
            parameters=parameters,
            protocol=dataset,
        )
        costs = []
        for cost in cost_class:
            if issubclass(cost, pybop.LogLikelihood):
                costs.append(cost(dataset, sigma0=self.sigma0))
            else:
                costs.append(cost(dataset))

        weighted_cost = pybop.WeightedCost(*costs, weights=[0.1, 1, 0.5])
        return pybop.Problem(simulator, weighted_cost)

    def test_fitting_costs(self, weighted_fitting_problem):
        x0 = weighted_fitting_problem.parameters.get_initial_values()
        options = pybop.PintsOptions(
            sigma=0.03,
            max_iterations=250,
            max_unchanged_iterations=35,
        )
        optim = pybop.CuckooSearch(problem=weighted_fitting_problem, options=options)

        initial_cost = optim.problem.evaluate(
            optim.problem.parameters.get_initial_values()
        )
        results = optim.run()

        # Assertions
        if not np.allclose(x0, self.ground_truth, atol=1e-5):
            if results.minimising:
                assert initial_cost > results.best_cost
            else:
                assert initial_cost < results.best_cost
        np.testing.assert_allclose(results.x, self.ground_truth, atol=1.5e-2)

    @pytest.fixture
    def design_targets(self):
        return [
            "Gravimetric energy density [Wh.kg-1]",
            "Volumetric energy density [Wh.m-3]",
        ]

    @pytest.fixture
    def weighted_design_cost(self, model, parameter_values, design_targets):
        initial_state = {"Initial SoC": 1.0}
        parameters = pybop.Parameters(
            pybop.Parameter(
                "Positive electrode thickness [m]",
                prior=pybop.Gaussian(5e-05, 5e-06),
                bounds=[2e-06, 10e-05],
            ),
            pybop.Parameter(
                "Negative electrode thickness [m]",
                prior=pybop.Gaussian(5e-05, 5e-06),
                bounds=[2e-06, 10e-05],
            ),
        )
        experiment = pybamm.Experiment(
            ["Discharge at 1C until 3.5 V (5 seconds period)"]
        )
        simulator = pybop.pybamm.Simulator(
            model,
            parameter_values=parameter_values,
            parameters=parameters,
            protocol=experiment,
            initial_state=initial_state,
            use_formation_concentrations=True,
        )
        costs = [pybop.DesignCost(target=target) for target in design_targets]
        cost = pybop.WeightedCost(*costs, weights=[1.0, 0.1])
        return pybop.Problem(simulator, cost)

    def test_design_costs(self, weighted_design_cost):
        problem = weighted_design_cost
        options = pybop.PintsOptions(max_iterations=15)
        optim = pybop.CuckooSearch(problem, options=options)
        initial_values = optim.problem.parameters.get_initial_values()
        initial_cost = optim.problem.evaluate(initial_values)
        results = optim.run()

        # Assertions
        assert initial_cost < results.best_cost
        for i, _ in enumerate(results.x):
            assert results.x[i] > initial_values[i]

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
