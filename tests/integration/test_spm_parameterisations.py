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
        return pybop.Parameters(
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
                # no bounds
            ),
        )

    @pytest.fixture(
        params=[
            pybop.GaussianLogLikelihoodKnownSigma,
            pybop.GaussianLogLikelihood,
            pybop.RootMeanSquaredError,
            pybop.MeanAbsoluteError,
            pybop.MeanSquaredError,
            pybop.SumSquaredError,
            pybop.SumOfPower,
            pybop.Minkowski,
            pybop.LogPosterior,
        ]
    )
    def cost_class(self, request):
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
    def optim(self, optimiser, model_and_parameter_values, parameters, cost_class):
        model, parameter_values = model_and_parameter_values
        parameter_values.set_initial_state(0.6)
        dataset = self.get_data(model, parameter_values)

        # Define the problem
        simulator = pybop.pybamm.Simulator(
            model,
            parameter_values=parameter_values,
            input_parameter_names=parameters.names,
            protocol=dataset,
        )

        # Construct the cost
        if cost_class is pybop.GaussianLogLikelihoodKnownSigma:
            cost = cost_class(dataset, sigma0=self.sigma0)
        elif cost_class is pybop.GaussianLogLikelihood:
            cost = cost_class(dataset, sigma0=self.sigma0 * 4)  # Initial sigma0 guess
        elif cost_class is pybop.LogPosterior:
            cost = cost_class(
                log_likelihood=pybop.GaussianLogLikelihoodKnownSigma(
                    dataset, sigma0=self.sigma0
                )
            )
        elif cost_class in [pybop.SumOfPower, pybop.Minkowski]:
            cost = cost_class(dataset, p=2.5)
        else:
            cost = cost_class(dataset)
        problem = pybop.Problem(simulator, parameters, cost)

        max_unchanged_iter = 100
        sigma0 = (
            [0.02, 0.02, 2e-3]
            if isinstance(cost, pybop.GaussianLogLikelihood)
            else 0.02
        )
        if optimiser is pybop.SimulatedAnnealing:
            max_unchanged_iter = 450
            sigma0 = [0.05, 0.05]
            if isinstance(cost, pybop.GaussianLogLikelihood):
                sigma0.append(2e-3)

        # Construct optimisation object
        if optimiser is pybop.SciPyDifferentialEvolution:
            options = pybop.SciPyDifferentialEvolutionOptions(maxiter=450)
        elif optimiser is pybop.SciPyMinimize:
            options = pybop.SciPyMinimizeOptions(maxiter=450)
        else:
            options = pybop.PintsOptions(
                max_iterations=450,
                max_unchanged_iterations=max_unchanged_iter,
                sigma=sigma0,
            )

        if optimiser in [
            pybop.SciPyDifferentialEvolution,
            pybop.CuckooSearch,
        ]:
            bounds = {"lower": [0.375, 0.375], "upper": [0.775, 0.775]}
            if isinstance(cost, pybop.GaussianLogLikelihood):
                bounds["lower"].append(0.0)
                bounds["upper"].append(0.05)
            problem.parameters.update(bounds=bounds)

        # Create optimiser
        optim = optimiser(problem, options=options)

        # Set Hypers
        if isinstance(optim, pybop.SimulatedAnnealing):
            optim.optimiser.cooling_rate = 0.85  # Cool quickly
        if isinstance(optim, pybop.CuckooSearch):
            optim.optimiser.pa = 0.55  # Increase abandon rate for limited iterations
        if isinstance(optim, pybop.AdamW):
            optim.optimiser.b1 = 0.9
            optim.optimiser.b2 = 0.9
        return optim

    def test_optimisers(self, optim):
        x0 = optim.problem.parameters.get_initial_values()

        # Add sigma0 to ground truth for GaussianLogLikelihood
        if isinstance(optim.problem.cost, pybop.GaussianLogLikelihood):
            self.ground_truth = np.concatenate(
                (self.ground_truth, np.asarray([self.sigma0]))
            )

        initial_cost = optim.problem(x0)
        results = optim.run()

        # Assertions
        if np.allclose(x0, self.ground_truth, atol=1e-5):
            raise AssertionError("Initial guess is too close to ground truth")

        if results.minimising:
            assert initial_cost > results.best_cost
        else:
            assert initial_cost < results.best_cost

        np.testing.assert_allclose(results.x, self.ground_truth, atol=1.5e-2)
        if isinstance(optim.problem.cost, pybop.GaussianLogLikelihood):
            np.testing.assert_allclose(results.x[-1], self.sigma0, atol=1e-3)

    @pytest.fixture
    def two_signal_problem(self, parameters, model_and_parameter_values, cost_class):
        model, parameter_values = model_and_parameter_values
        parameter_values.set_initial_state(0.5)
        dataset = self.get_data(model, parameter_values, include_ocv=True)

        # Define the cost to optimise
        target = ["Voltage [V]", "Bulk open-circuit voltage [V]"]
        simulator = pybop.pybamm.Simulator(
            model,
            parameter_values=parameter_values,
            input_parameter_names=parameters.names,
            protocol=dataset,
        )

        # Construct the cost
        if cost_class is pybop.GaussianLogLikelihoodKnownSigma:
            cost = cost_class(dataset, target=target, sigma0=self.sigma0)
        elif cost_class is pybop.GaussianLogLikelihood:
            cost = cost_class(
                dataset, target=target, sigma0=self.sigma0 * 4
            )  # Initial sigma0 guess
        elif cost_class is pybop.LogPosterior:
            cost = cost_class(
                log_likelihood=pybop.GaussianLogLikelihoodKnownSigma(
                    dataset, target=target, sigma0=self.sigma0
                )
            )
        elif cost_class in [pybop.SumOfPower, pybop.Minkowski]:
            cost = cost_class(dataset, target=target, p=2.5)
        else:
            cost = cost_class(dataset, target=target)
        return pybop.Problem(simulator, parameters, cost)

    @pytest.mark.parametrize(
        "multi_optimiser",
        [
            pybop.SciPyDifferentialEvolution,
            pybop.IRPropMin,
            pybop.CMAES,
        ],
    )
    def test_multiple_signals(self, multi_optimiser, two_signal_problem):
        x0 = two_signal_problem.parameters.get_initial_values()
        combined_sigma0 = np.asarray([self.sigma0, self.sigma0])

        if multi_optimiser is pybop.SciPyDifferentialEvolution:
            options = pybop.SciPyDifferentialEvolutionOptions(maxiter=250)
        elif multi_optimiser is pybop.SciPyMinimize:
            options = pybop.SciPyMinimizeOptions(maxiter=450)
        else:
            options = pybop.PintsOptions(
                max_iterations=250,
                max_unchanged_iterations=60,
                sigma=[0.03, 0.03, 6e-3, 6e-3]
                if isinstance(two_signal_problem.cost, pybop.GaussianLogLikelihood)
                else 0.03,
            )

        if multi_optimiser is pybop.SciPyDifferentialEvolution:
            bounds = {"lower": [0.375, 0.375], "upper": [0.775, 0.775]}
            if isinstance(two_signal_problem.cost, pybop.GaussianLogLikelihood):
                bounds["lower"].extend([0.0, 0.0])
                bounds["upper"].extend([0.05, 0.05])
            two_signal_problem.parameters.update(bounds=bounds)

        # Test each optimiser
        optim = multi_optimiser(two_signal_problem, options=options)

        # Add sigma0 to ground truth for GaussianLogLikelihood
        if isinstance(two_signal_problem.cost, pybop.GaussianLogLikelihood):
            self.ground_truth = np.concatenate((self.ground_truth, combined_sigma0))

        initial_cost = optim.problem(optim.problem.parameters.get_initial_values())
        results = optim.run()

        # Assertions
        if np.allclose(x0, self.ground_truth, atol=1e-5):
            raise AssertionError("Initial guess is too close to ground truth")

        if results.minimising:
            assert initial_cost > results.best_cost
        else:
            assert initial_cost < results.best_cost

        np.testing.assert_allclose(results.x, self.ground_truth, atol=1.5e-2)
        if isinstance(two_signal_problem.cost, pybop.GaussianLogLikelihood):
            np.testing.assert_allclose(results.x[-2:], combined_sigma0, atol=5e-4)

    @pytest.mark.parametrize("init_soc", [0.4, 0.6])
    def test_model_misparameterisation(
        self, parameters, model_and_parameter_values, init_soc
    ):
        model, parameter_values = model_and_parameter_values
        parameter_values.set_initial_state(init_soc)

        # Define two different models with different parameter sets
        # The optimisation should fail as the models are not the same
        second_parameter_values = pybamm.ParameterValues("Ecker2015")
        second_parameter_values.set_initial_state(init_soc)
        second_model = pybamm.lithium_ion.SPMe()
        simulator = pybop.pybamm.Simulator(
            second_model, parameter_values=second_parameter_values
        )
        dataset = self.get_data(second_model, second_parameter_values)

        # Define the cost to optimise
        simulator = pybop.pybamm.Simulator(
            model,
            parameter_values=parameter_values,
            input_parameter_names=parameters.names,
            protocol=dataset,
        )
        cost = pybop.RootMeanSquaredError(dataset)
        problem = pybop.Problem(simulator, parameters, cost)

        # Build the optimisation problem
        optim = pybop.XNES(problem)
        initial_cost = problem(parameters.get_initial_values())

        # Run the optimisation problem
        results = optim.run()

        # Assertion for best_cost
        assert initial_cost > results.best_cost

        # Assertion for x
        with np.testing.assert_raises(AssertionError):
            np.testing.assert_allclose(results.x, self.ground_truth, atol=2e-2)

    def get_data(self, model, parameter_values, include_ocv=False):
        experiment = pybamm.Experiment(
            [
                "Discharge at 0.5C for 8 minutes (8 second period)",
                "Charge at 0.5C for 8 minutes (8 second period)",
            ]
        )
        solution = pybamm.Simulation(
            model, parameter_values=parameter_values, experiment=experiment
        ).solve()
        if not include_ocv:
            return pybop.Dataset(
                {
                    "Time [s]": solution["Time [s]"].data,
                    "Current function [A]": solution["Current [A]"].data,
                    "Voltage [V]": self.noisy(
                        solution["Voltage [V]"].data, self.sigma0
                    ),
                }
            )
        return pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": self.noisy(solution["Voltage [V]"].data, self.sigma0),
                "Bulk open-circuit voltage [V]": self.noisy(
                    solution["Bulk open-circuit voltage [V]"].data, self.sigma0
                ),
            }
        )
