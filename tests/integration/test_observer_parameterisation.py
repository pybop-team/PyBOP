import numpy as np
import pytest
from pybamm import IDAKLUSolver

import pybop


class TestObservers:
    """
    A class to run integration tests on the Observers class.
    """

    pytestmark = pytest.mark.integration

    @pytest.fixture(autouse=True)
    def setup(self):
        self.ground_truth = np.clip(
            np.array([0.1, 1.0]) + np.random.normal(loc=0.0, scale=0.05, size=2),
            a_min=[0.04, 0.2],
            a_max=[0.85, 1.15],
        )

    @pytest.fixture
    def parameter_set(self):
        return pybop.ParameterSet.to_pybamm(
            {"k": self.ground_truth[0], "y0": self.ground_truth[1]}
        )

    @pytest.fixture
    def model(self, parameter_set):
        return pybop.ExponentialDecayModel(
            parameter_set=parameter_set, solver=IDAKLUSolver, n_states=1
        )

    @pytest.fixture
    def parameters(self, parameter_set):
        return pybop.Parameters(
            pybop.Parameter(
                "k",
                prior=pybop.Gaussian(0.1, 0.05),
                bounds=[0, 1],
                true_value=parameter_set["k"],
            ),
            pybop.Parameter(
                "y0",
                prior=pybop.Gaussian(1, 0.05),
                bounds=[0, 3],
                true_value=parameter_set["y0"],
            ),
        )

    def noise(self, sigma, values):
        return np.random.normal(0, sigma, values)

    def test_observer_exponential_decay(self, parameters, model):
        # Make a prediction with measurement noise
        sigma = 1e-2
        t_eval = np.linspace(0, 20, 10)
        values = model.predict(t_eval=t_eval)["y_0"].data
        corrupt_values = values + self.noise(sigma, len(t_eval))

        # Form dataset
        dataset = pybop.Dataset(
            {
                "Time [s]": t_eval,
                "Current function [A]": 0 * t_eval,  # placeholder
                "y_0": corrupt_values,
            }
        )

        # Define the UKF observer
        signal = ["y_0"]
        n_states = model.n_states
        n_signals = len(signal)
        covariance = np.diag([sigma**2] * n_states)
        process_noise = np.diag([1e-6] * n_states)
        measurement_noise = np.diag([sigma**2] * n_signals)
        observer = pybop.UnscentedKalmanFilterObserver(
            parameters,
            model,
            covariance,
            process_noise,
            measurement_noise,
            dataset,
            signal=signal,
        )

        # Generate cost function, and optimisation class
        cost = pybop.ObserverCost(observer)
        optim = pybop.CMAES(cost, max_unchanged_iterations=25, verbose=True)

        # Initial Cost
        x0 = cost.parameters.initial_value()
        initial_cost = optim.cost(x0)

        # Run optimisation
        results = optim.run()

        # Assertions
        if not np.allclose(x0, self.ground_truth, atol=1e-5):
            if results.minimising:
                assert initial_cost > results.final_cost
            else:
                assert initial_cost < results.final_cost
        else:
            raise ValueError("Initial value is the same as the ground truth value.")
        np.testing.assert_allclose(results.x, parameters.true_value(), atol=1.5e-2)
