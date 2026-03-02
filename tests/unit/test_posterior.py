import numpy as np
import pybamm
import pytest

import pybop


class TestLogPosterior:
    """
    Class for log posterior unit tests
    """

    pytestmark = pytest.mark.unit

    @pytest.fixture
    def model(self):
        return pybamm.lithium_ion.SPM()

    @pytest.fixture
    def ground_truth(self):
        return 0.52

    @pytest.fixture
    def parameter_values(self, model, ground_truth):
        parameter_values = model.default_parameter_values
        parameter_values.update(
            {"Negative electrode active material volume fraction": ground_truth}
        )
        return parameter_values

    @pytest.fixture
    def parameter(self, ground_truth):
        return {
            "Negative electrode active material volume fraction": pybop.Parameter(
                distribution=pybop.Gaussian(0.5, 0.01, truncated_at=[0.375, 0.625]),
                initial_value=ground_truth,
            )
        }

    @pytest.fixture
    def experiment(self):
        return pybamm.Experiment(["Discharge at 1C for 1 minutes (5 second period)"])

    @pytest.fixture
    def dataset(self, model, parameter_values, experiment):
        solution = pybamm.Simulation(
            model, parameter_values=parameter_values, experiment=experiment
        ).solve()
        return pybop.import_pybamm_solution(solution)

    @pytest.fixture
    def simulator(self, model, parameter_values, parameter, dataset):
        parameter_values.update(parameter)
        return pybop.pybamm.Simulator(
            model, parameter_values=parameter_values, protocol=dataset
        )

    @pytest.fixture
    def log_pdf(self, dataset, simulator):
        cost = pybop.GaussianLogLikelihoodKnownSigma(dataset, sigma=0.01)
        return pybop.LogPosterior(simulator, cost)

    def test_log_pdf(self, log_pdf):
        log_pdf.parameters["Negative electrode active material volume fraction"] = (
            pybop.Parameter(distribution=pybop.Gaussian(0.5, 0.01))
        )
        # Test log posterior
        x = np.array([0.50])
        assert np.allclose(log_pdf.evaluate(x).values, 51.6033, atol=2e-2)

        # Test log posterior evaluateS1
        p, dp = log_pdf.evaluate(x, calculate_sensitivities=True).get_values()
        assert np.allclose(p, 51.6033, atol=2e-2)
        assert np.allclose(dp[log_pdf.parameters.names[0]], 0.4266, atol=2e-2)

    def test_log_pdf_inf(self, log_pdf):
        log_pdf.parameters["Negative electrode active material volume fraction"] = (
            pybop.Parameter(distribution=pybop.Uniform(0.45, 0.55))
        )

        # Test prior np.inf
        assert not np.isfinite(log_pdf.evaluate([1]).values)
        assert not np.isfinite(
            log_pdf.evaluate([1], calculate_sensitivities=True).values
        )
