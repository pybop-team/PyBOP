import numpy as np
import pybamm
import pytest

import pybop


class Test_SPM_Parameterisation:
    """
    A class to test the model parameterisation methods.
    """

    pytestmark = pytest.mark.integration

    @pytest.fixture(autouse=True)
    def setup(self):
        self.sigma0 = 0.002
        self.ground_truth = np.clip(
            np.asarray([0.55, 0.55]) + np.random.normal(loc=0.0, scale=0.05, size=2),
            a_min=0.425,
            a_max=0.75,
        )

    @pytest.fixture
    def model_and_parameter_values(self):
        model = pybamm.lithium_ion.SPM()

        # Use stoichiometry rather than voltage-based events
        var = model.variables
        model.events = [
            pybamm.Event(
                "Minimum negative particle surface stoichiometry",
                pybamm.min(var["Negative particle surface stoichiometry"]) - 0.01,
            ),
            pybamm.Event(
                "Maximum negative particle surface stoichiometry",
                (1 - 0.01) - pybamm.max(var["Negative particle surface stoichiometry"]),
            ),
            pybamm.Event(
                "Minimum positive particle surface stoichiometry",
                pybamm.min(var["Positive particle surface stoichiometry"]) - 0.01,
            ),
            pybamm.Event(
                "Maximum positive particle surface stoichiometry",
                (1 - 0.01) - pybamm.max(var["Positive particle surface stoichiometry"]),
            ),
        ]

        parameter_values = pybamm.ParameterValues("Chen2020")
        x = self.ground_truth
        parameter_values.update(
            {
                "Negative electrode active material volume fraction": x[0],
                "Positive electrode active material volume fraction": x[1],
            }
        )

        # Fix the total lithium concentration to simplify the fitting problem
        model.param.Q_Li_particles_init = parameter_values.evaluate(
            model.param.Q_Li_particles_init
        )
        return model, parameter_values

    @pytest.fixture
    def parameters(self):
        return [
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Uniform(0.3, 0.9),
                initial_value=pybop.Uniform(0.4, 0.75).rvs()[0],
                bounds=[0.3, 0.8],
            ),
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                prior=pybop.Uniform(0.3, 0.9),
                initial_value=pybop.Uniform(0.4, 0.75).rvs()[0],
                bounds=[0.3, 0.8],
            ),
        ]

    @pytest.fixture(
        params=[
            pybop.costs.pybamm.RootMeanSquaredError,
            pybop.costs.pybamm.MeanAbsoluteError,
            pybop.costs.pybamm.MeanSquaredError,
            pybop.costs.pybamm.SumSquaredError,
            pybop.costs.pybamm.SumOfPower,
            pybop.costs.pybamm.Minkowski,
        ]
    )
    def cost_cls(self, request):
        return request.param

    def noisy(self, data, sigma):
        return data + np.random.normal(0, sigma, len(data))

    @pytest.fixture(
        params=[
            pybop.SciPyMinimize,  # scipy with sensitivities
            pybop.SciPyDifferentialEvolution,  # without sensitivites
            pybop.IRPropMin,  # pints optimiser with sensitivities
            pybop.NelderMead,  # pints without sensitivities
        ]
    )
    def optimiser(self, request):
        return request.param

    @pytest.fixture
    def experiment(self):
        return pybamm.Experiment(
            [
                "Rest for 1 second",
                "Discharge at 0.5C for 8 minutes (8 second period)",
                "Charge at 0.5C for 8 minutes (8 second period)",
            ]
        )

    @pytest.fixture
    def dataset(self, model_and_parameter_values, experiment):
        model, parameter_values = model_and_parameter_values
        sim = pybamm.Simulation(
            model=model,
            parameter_values=parameter_values,
            experiment=experiment,
        )
        sol = sim.solve()
        _, mask = np.unique(sol.t, return_index=True)
        return pybop.Dataset(
            {
                "Time [s]": sol.t[mask],
                "Current function [A]": sol["Current [A]"].data[mask],
                "Voltage [V]": self.noisy(sol["Voltage [V]"].data[mask], self.sigma0),
            }
        )

    @pytest.fixture
    def problem(self, model_and_parameter_values, parameters, cost_cls, dataset):
        model, parameter_values = model_and_parameter_values
        builder = pybop.Pybamm()
        builder.set_simulation(model, parameter_values=parameter_values)
        builder.set_dataset(dataset)
        for p in parameters:
            builder.add_parameter(p)
        signal = "Voltage [V]"
        if cost_cls is pybop.costs.pybamm.NegativeGaussianLogLikelihood:
            cost = cost_cls(signal, signal)
        elif cost_cls in [pybop.SumOfPower, pybop.Minkowski]:
            cost = cost_cls(signal, signal, p=2.5)
        else:
            cost = cost_cls(signal, signal)
        builder.add_cost(cost)
        problem = builder.build()
        return problem

    def test_problem(self, problem):
        problem.set_params(self.ground_truth)
        cost_at_ground = problem.run()
        ground_plus_delta = self.ground_truth + np.random.normal(
            0, 0.1, len(self.ground_truth)
        )
        problem.set_params(ground_plus_delta)
        cost_at_ground_plus_delta = problem.run()
        assert cost_at_ground < cost_at_ground_plus_delta

    @pytest.fixture
    def optim(self, optimiser, problem):
        options = optimiser.default_options()
        if issubclass(optimiser, pybop.BaseSciPyOptimiser):
            options.maxiter = 100
            options.atol = 1e-6
        else:
            options.max_iterations = 100
            options.absolute_tolerance = 1e-6
            options.max_unchanged_iterations = 30
            options.sigma = 2e-2

        # Set sigma0 and create optimiser
        return optimiser(problem, options=options)

    def test_optimisers(self, optim, cost_cls):
        x0 = optim.problem.params.get_initial_values()

        # Add sigma0 to ground truth for GaussianLogLikelihood
        if cost_cls in (pybop.costs.pybamm.NegativeGaussianLogLikelihood,):
            self.ground_truth = np.concatenate(
                (self.ground_truth, np.asarray([self.sigma0]))
            )

        results = optim.run()

        # Assertions
        if np.allclose(x0, self.ground_truth, atol=1e-5):
            raise AssertionError("Initial guess is too close to ground truth")

        assert results.initial_cost > results.best_cost

        np.testing.assert_allclose(results.x, self.ground_truth, atol=1.5e-2)

    def test_with_init_soc(self, model_and_parameter_values, parameters, experiment):
        model, parameter_values = model_and_parameter_values
        init_soc = 0.6
        sim = pybamm.Simulation(
            model=model,
            parameter_values=parameter_values,
            experiment=experiment,
        )
        sol = sim.solve(initial_soc=init_soc)
        _, mask = np.unique(sol.t, return_index=True)
        dataset = pybop.Dataset(
            {
                "Time [s]": sol.t[mask],
                "Current function [A]": sol["Current [A]"].data[mask],
                "Voltage [V]": self.noisy(sol["Voltage [V]"].data[mask], self.sigma0),
            }
        )
        builder = pybop.Pybamm()
        builder.set_simulation(
            model,
            parameter_values=parameter_values,
            initial_state=f"{sol['Voltage [V]'].data[0]} V",
        )
        builder.set_dataset(dataset)
        for p in parameters:
            builder.add_parameter(p)
        signal = "Voltage [V]"
        builder.add_cost(pybop.costs.pybamm.SumSquaredError(signal, signal))
        problem = builder.build()
        options = pybop.PintsOptions(
            max_iterations=100,
            absolute_tolerance=1e-6,
            max_unchanged_iterations=30,
        )
        optim = pybop.NelderMead(problem, options)

        x0 = optim.problem.params.get_initial_values()
        results = optim.run()
        # Assertions
        if np.allclose(x0, self.ground_truth, atol=1e-5):
            raise AssertionError("Initial guess is too close to ground truth")

        assert results.initial_cost > results.best_cost
        np.testing.assert_allclose(results.x, self.ground_truth, atol=1.5e-2)
