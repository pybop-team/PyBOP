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
    def model(self):
        return pybamm.lithium_ion.SPM()

    @pytest.fixture
    def parameter_values(self):
        parameter_values = pybamm.ParameterValues("Chen2020")
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
            pybop.SciPyDifferentialEvolution,
            pybop.SimulatedAnnealing,
            pybop.CuckooSearch,
            pybop.NelderMead,
            pybop.IRPropMin,
            pybop.IRPropPlus,
            pybop.AdamW,
            pybop.CMAES,
            pybop.SNES,
            pybop.XNES,
        ]
    )
    def optimiser(self, request):
        return request.param

    @pytest.fixture
    def dataset(self, model, parameter_values):
        experiment = pybamm.Experiment(
            [
                "Rest for 1 second",
                "Discharge at 0.5C for 8 minutes (8 second period)",
                "Charge at 0.5C for 8 minutes (8 second period)",
            ]
        )
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
    def problem(self, model, parameters, cost_cls, parameter_values, dataset):
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
        options.max_iterations = 100
        if isinstance(options, pybop.SciPyDifferentialEvolutionOptions):
            options.atol = 1e-6
        elif isinstance(options, pybop.PintsOptions):
            options.absolute_tolerance = 1e-6
            options.max_unchanged_iterations = 30
            options.sigma = 2e-2

        # Set sigma0 and create optimiser
        optim = optimiser(problem, options=options)
        if isinstance(optim, pybop.SimulatedAnnealing):
            optim.optimiser.cooling_rate = 0.8  # Cool quickly
            optim.set_max_unchanged_iterations(50)
        return optim

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

    def test_with_init_soc(self, model, parameters, parameter_values):
        experiment = pybamm.Experiment(
            [
                "Rest for 2 seconds",
                "Discharge at 0.5C for 8 minutes (8 second period)",
                "Charge at 0.5C for 8 minutes (8 second period)",
            ]
        )
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
