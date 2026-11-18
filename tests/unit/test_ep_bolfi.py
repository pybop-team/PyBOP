import numpy as np
import pybamm
import pytest

import pybop


class TestEP_BOLFI:
    """
    A class to test the EP-BOLFI optimiser. Skip if the ep_bolfi package is not available.
    """

    pytestmark = pytest.mark.unit
    pytest.importorskip("ep_bolfi")

    @pytest.fixture
    def dataset(self):
        return pybop.Dataset(
            {
                "Time [s]": np.linspace(0, 360, 10),
                "Current [A]": 1e-2 * np.ones(10),
                "Voltage [V]": np.ones(10),
            }
        )

    @pytest.fixture
    def model(self):
        return pybamm.lithium_ion.SPM()

    @pytest.fixture
    def multivariate_simulator(self, model, dataset):
        parameter_values = model.default_parameter_values
        distribution = pybop.MultivariateGaussian(
            mean=[0.6, 0.5], covariance=[[0.02, 0.0], [0.0, 0.05]]
        )
        parameter_values.update(
            {
                "Negative electrode active material volume fraction": pybop.Parameter(
                    distribution=pybop.MarginalDistribution(distribution, 0),
                    initial_value=0.6,
                ),
                "Positive electrode active material volume fraction": pybop.Parameter(
                    distribution=pybop.MarginalDistribution(distribution, 1),
                    initial_value=0.5,
                ),
            }
        )
        simulator = pybop.pybamm.Simulator(
            model, parameter_values=parameter_values, protocol=dataset
        )

        return simulator

    @pytest.fixture
    def multivariate_problem(self, multivariate_simulator, dataset):
        cost = pybop.SquareRootFeatureDistance(dataset, feature="offset")
        return pybop.Problem(multivariate_simulator, cost)

    @pytest.fixture
    def gitt_like_problem(self, multivariate_simulator, dataset):
        sqrt_cost_1 = pybop.costs.feature_distances.SquareRootFeatureDistance(
            dataset,
            target="Voltage [V]",
            feature="offset",
            time_start=0,
            time_end=180,
        )
        sqrt_cost_2 = pybop.costs.feature_distances.SquareRootFeatureDistance(
            dataset,
            target="Voltage [V]",
            feature="offset",
            time_start=180,
            time_end=360,
        )
        problem = pybop.MetaProblem(
            pybop.Problem(multivariate_simulator, sqrt_cost_1),
            pybop.Problem(multivariate_simulator, sqrt_cost_2),
        )
        return problem

    def test_ep_bolfi(self, multivariate_problem, gitt_like_problem):
        options = pybop.EPBOLFIOptions()
        optim = pybop.EP_BOLFI(multivariate_problem, options=options)
        result = optim.run()
        assert result.posterior is not None
        assert (
            optim.name()
            == "Expectation Propagation with Bayesian Optimization for Likelihood-Free Inference"
        )
        with pytest.raises(
            ValueError,
            match="EP-BOLFI is not parallelisable by design for sample efficiency. Use SOBER instead for parallelisation.",
        ):
            optim = pybop.EP_BOLFI(
                multivariate_problem, options=pybop.EPBOLFIOptions(parallel=True)
            )
        with pytest.raises(
            ValueError,
            match="The EP dampening has to be a positive number smaller than 1.",
        ):
            optim = pybop.EP_BOLFI(
                multivariate_problem, options=pybop.EPBOLFIOptions(ep_total_dampening=1)
            )
        with pytest.raises(
            ValueError,
            match="Hard parameter boundaries can't be negative multiples of σ.",
        ):
            optim = pybop.EP_BOLFI(
                multivariate_problem,
                options=pybop.EPBOLFIOptions(boundaries_in_standard_deviations=-1),
            )
        with pytest.raises(
            ValueError,
            match="Initial Sobol parameter samples can not be a negative number.",
        ):
            optim = pybop.EP_BOLFI(
                multivariate_problem,
                options=pybop.EPBOLFIOptions(bolfi_initial_sobol_samples=-1),
            )
        with pytest.raises(
            ValueError,
            match="Optimally acquired parameter samples can not be a negative number.",
        ):
            optim = pybop.EP_BOLFI(
                multivariate_problem,
                options=pybop.EPBOLFIOptions(bolfi_optimally_acquired_samples=-1),
            )
        with pytest.raises(
            ValueError,
            match="Effective Sample Size for posterior evaluation can not be a negative number.",
        ):
            optim = pybop.EP_BOLFI(
                multivariate_problem,
                options=pybop.EPBOLFIOptions(bolfi_posterior_effective_sample_size=-1),
            )
        with pytest.raises(
            ValueError,
            match="The factor by which to increase posterior samples has to be greater than 1.",
        ):
            optim = pybop.EP_BOLFI(
                multivariate_problem,
                options=pybop.EPBOLFIOptions(posterior_actual_sample_size_increase=1),
            )
        with pytest.raises(
            ValueError,
            match="The factor by which to increase model samples has to be greater than 1.",
        ):
            optim = pybop.EP_BOLFI(
                multivariate_problem,
                options=pybop.EPBOLFIOptions(posterior_model_resample_size_increase=1),
            )
        with pytest.raises(
            ValueError,
            match="The Gelman-Rubin threshold has to be a number greater than 1.",
        ):
            optim = pybop.EP_BOLFI(
                multivariate_problem,
                options=pybop.EPBOLFIOptions(posterior_gelman_rubin_threshold=1),
            )
        options = pybop.EPBOLFIOptions()
        optim = pybop.EP_BOLFI(gitt_like_problem, options=options)
        result = optim.run()
        assert result.posterior is not None
