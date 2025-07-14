import numpy as np
import pybamm
import pytest

import pybop
import pybop.pipelines._pybamm_eis_pipeline


class TestEISParameterisation:
    """
    A class to test the eis parameterisation methods.
    """

    pytestmark = pytest.mark.integration

    @pytest.fixture(autouse=True)
    def setup(self):
        self.sigma0 = 5e-4
        self.ground_truth = np.clip(
            np.asarray([0.55, 0.55]) + np.random.normal(loc=0.0, scale=0.05, size=2),
            a_min=0.4,
            a_max=0.75,
        )

    @pytest.fixture
    def parameter_values(self):
        params = pybamm.ParameterValues("Chen2020")
        x = self.ground_truth
        params.update(
            {
                "Negative electrode active material volume fraction": x[0],
                "Positive electrode active material volume fraction": x[1],
            }
        )
        return params

    @pytest.fixture
    def model(self):
        return pybamm.lithium_ion.SPM(
            options={"surface form": "differential"},
        )

    @pytest.fixture
    def parameters(self):
        return pybop.Parameters(
            [
                pybop.Parameter(
                    "Negative electrode active material volume fraction",
                    prior=pybop.Uniform(0.3, 0.9),
                    initial_value=pybop.Uniform(0.4, 0.75).rvs()[0],
                    bounds=[0.375, 0.775],
                ),
                pybop.Parameter(
                    "Positive electrode active material volume fraction",
                    prior=pybop.Uniform(0.3, 0.9),
                    initial_value=pybop.Uniform(0.4, 0.75).rvs()[0],
                    bounds=[0.375, 0.775],
                ),
            ]
        )

    @pytest.fixture(params=[0.5])
    def init_soc(self, request):
        return request.param

    @pytest.fixture(
        params=[
            pybop.SumSquaredError,
            pybop.MeanAbsoluteError,
            pybop.MeanSquaredError,
            pybop.Minkowski,
        ]
    )
    def cost(self, request):
        return request.param

    def noisy(self, data, sigma):
        # Generate real part noise
        real_noise = np.random.normal(0, sigma, len(data))

        # Generate imaginary part noise
        imag_noise = np.random.normal(0, sigma, len(data))

        # Combine them into a complex noise
        return data + real_noise + 1j * imag_noise

    @pytest.fixture(
        params=[
            pybop.ScipyDifferentialEvolution,
            pybop.CMAES,
            pybop.CuckooSearch,
            pybop.XNES,
        ]
    )
    def optimiser(self, request):
        return request.param

    @pytest.fixture
    def dataset(self, model, parameters, init_soc, parameter_values):
        n_frequency = 15
        # Set frequency set
        f_eval = np.logspace(-4, 5, n_frequency)

        dummy_dataset = pybop.Dataset(
            {
                "Frequency [Hz]": f_eval,
                "Current function [A]": np.ones(n_frequency) * 0.0,
                "Impedance": np.ones(n_frequency) * 0.0,
            }
        )
        builder = pybop.PybammEIS()
        builder.set_simulation(model, parameter_values=parameter_values)
        builder.set_dataset(dummy_dataset)
        for p in parameters:
            builder.add_parameter(p)
        problem = builder.build()
        sol = problem.simulate(parameters.to_dict(self.ground_truth))

        return pybop.Dataset(
            {
                "Frequency [Hz]": f_eval,
                "Current function [A]": np.ones(n_frequency) * 0.0,
                "Impedance": self.noisy(sol, self.sigma0),
                "Impedance No Noise": sol,
            }
        )

    @pytest.fixture
    def problem(self, model, parameters, cost, init_soc, parameter_values, dataset):
        builder = pybop.PybammEIS()
        builder.set_simulation(model, parameter_values=parameter_values)
        builder.set_dataset(dataset)
        for p in parameters:
            builder.add_parameter(p)
        builder.add_cost(cost())
        return builder.build()

    @pytest.fixture
    def optim(self, optimiser, problem):
        options = optimiser.default_options()
        if isinstance(options, pybop.ScipyDifferentialEvolutionOptions):
            options.max_iterations = 100
            options.atol = 1e-6
        elif isinstance(options, pybop.PintsOptions):
            options.max_unchanged_iterations = 35
            options.max_iterations = 100
            options.absolute_tolerance = 1e-6

        # Create optimiser
        return optimiser(problem, options=options)

    def test_eis_optimisers(self, optim, dataset, parameters):
        sol = optim.problem.simulate(parameters.to_dict(self.ground_truth))

        # Check that the simulated impedance matches the dataset impedance
        np.testing.assert_allclose(
            sol,
            dataset["Impedance No Noise"],
            atol=1e-5,
            err_msg="Simulated impedance does not match dataset impedance",
        )
        x0 = optim.problem.params.get_initial_values()

        optim.problem.set_params(x0)
        initial_cost = optim.problem.run()
        results = optim.run()

        # Assertions
        if np.allclose(x0, self.ground_truth, atol=1e-5):
            raise AssertionError("Initial guess is too close to ground truth")

        # Assert on identified values, without sigma for GaussianLogLikelihood
        # as the sigma values are small (5e-4), this is a difficult identification process
        # and requires a high number of iterations, and parameter dependent step sizes.
        assert initial_cost > results.final_cost
        np.testing.assert_allclose(results.x, self.ground_truth, atol=1.5e-2)
