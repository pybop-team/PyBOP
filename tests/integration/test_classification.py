import json

import numpy as np
import pybamm
import pytest
from pybamm import Parameter

import pybop


class TestClassification:
    """
    A class to test the classification of different optimisation results.
    """

    pytestmark = pytest.mark.integration

    @pytest.fixture(
        params=[
            np.asarray([0.05, 0.05]),
            np.asarray([0.1, 0.05]),
            np.asarray([0.05, 0.01]),
        ]
    )
    def parameters(self, request):
        self.ground_truth = request.param
        return {
            "R0 [Ohm]": pybop.Parameter(
                prior=pybop.Gaussian(0.05, 0.01),
                bounds=[0.02, 0.08],
            ),
            "R1 [Ohm]": pybop.Parameter(
                prior=pybop.Gaussian(0.05, 0.01),
                bounds=[0.02, 0.08],
            ),
        }

    @pytest.fixture
    def model(self):
        return pybamm.equivalent_circuit.Thevenin()

    @pytest.fixture
    def parameter_values(self, model, parameters):
        with open("examples/parameters/initial_ecm_parameters.json") as file:
            parameter_values = pybamm.ParameterValues(json.load(file))
        parameter_values.update(
            {
                "Open-circuit voltage [V]": model.default_parameter_values[
                    "Open-circuit voltage [V]"
                ]
            },
            check_already_exists=False,
        )
        parameter_values.update({"C1 [F]": 1000})
        parameter_values.update(
            {"R0 [Ohm]": self.ground_truth[0], "R1 [Ohm]": self.ground_truth[1]}
        )
        return parameter_values

    @pytest.fixture
    def dataset(self, model, parameter_values):
        experiment = pybamm.Experiment(
            [
                "Discharge at 0.5C for 2 minutes (4 seconds period)",
                "Charge at 0.5C for 2 minutes (4 seconds period)",
            ]
        )
        solution = pybamm.Simulation(
            model, parameter_values=parameter_values, experiment=experiment
        ).solve()
        return pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": solution["Voltage [V]"].data,
            }
        )

    @pytest.fixture
    def simulator(self, model, parameter_values, parameters, dataset):
        parameter_values.update(parameters)
        return pybop.pybamm.Simulator(
            model, parameter_values=parameter_values, protocol=dataset
        )

    def test_classify_using_hessian(self, simulator, dataset):
        cost = pybop.RootMeanSquaredError(dataset)
        problem = pybop.Problem(simulator, cost)
        x = self.ground_truth
        bounds = problem.parameters.get_bounds()
        x0 = np.clip(x, bounds["lower"], bounds["upper"])
        optim = pybop.XNES(problem)
        logger = pybop.Logger(minimising=problem.minimising)
        logger.iteration = 1
        logger.extend_log(x_search=[x0], x_model=[x0], cost=[problem(x0)])
        result = pybop.OptimisationResult(optim=optim, logger=logger, time=1.0)

        message, info = pybop.classify_using_hessian(result)

        assert isinstance(info, dict)
        for k in (
            "hessian_fd",
            "eigenvalues",
            "eigenvectors",
            "x",
            "dx",
            "names",
            "best_cost",
            "span0",
            "span1",
            "param0",
            "param1",
            "Z",
        ):
            assert k in info
        assert "hessian_fd" in info and info["hessian_fd"].shape == (2, 2)
        assert "eigenvalues" in info and info["eigenvalues"].shape == (2,)
        assert "eigenvectors" in info and info["eigenvectors"].shape == (2, 2)
        assert info["x"].shape == (2,)
        assert "Z" in info and info["Z"].ndim == 2
        assert info["Z"].shape[0] == info["Z"].shape[1]  # grid is square

        # Hessian may be NaN on some model combinations
        H = info["hessian_fd"]

        if np.isfinite(H).all():
            # finite and (approximately) symmetric
            assert np.allclose(H, H.T, atol=1e-8)
            # Eigenvalues should be sorted ascending (only meaningful if finite)
            evals = info["eigenvalues"]
            assert evals[0] <= evals[1]
            # Eigenvectors should be finite and non-zero norm
            evecs = info["eigenvectors"]
            assert np.isfinite(evecs).all()
            for k in range(evecs.shape[1]):
                nrm = np.linalg.norm(evecs[:, k])
                assert nrm > 0.0
        else:
            # If full Hessian contains NaNs, try to use eigenvalues/eigenvectors
            evals = info.get("eigenvalues", None)
            evecs = info.get("eigenvectors", None)

            if evals is not None and np.all(np.isfinite(evals)):
                # If eigenvalues are finite we can still check ordering
                assert evals[0] <= evals[1]

            elif evecs is not None and np.isfinite(evecs).all():
                # If eigenvectors are finite, check their norms
                for k in range(evecs.shape[1]):
                    nrm = np.linalg.norm(evecs[:, k])
                    assert nrm > 0.0

            else:
                # Fallback: ensure the Z grid exists (this confirms classification ran)
                assert isinstance(info.get("Z", None), np.ndarray)

        # Check the grid Z contains at least some finite values
        Z = info["Z"]
        assert np.isfinite(Z).any()

        # Check p0 and p1
        p0 = info["param0"]
        p1 = info["param1"]
        x_center = info.get("x", np.asarray(x))
        assert p0.min() < x_center[0] < p0.max()
        assert p1.min() < x_center[1] < p1.max()

        if np.all(x == np.asarray([0.05, 0.05])):
            message, _ = pybop.classify_using_hessian(result)
            assert message == "The optimiser has located a minimum."
        elif np.all(x == np.asarray([0.1, 0.05])):
            message, _ = pybop.classify_using_hessian(result)
            assert message == (
                "The optimiser has not converged to a stationary point."
                " The result is near the upper bound of R0 [Ohm]."
            )
        elif np.all(x == np.asarray([0.05, 0.01])):
            message, _ = pybop.classify_using_hessian(result)
            assert message == (
                "The optimiser has not converged to a stationary point."
                " The result is near the lower bound of R1 [Ohm]."
            )
        else:
            raise Exception(f"Please add a check for these values: {x}")

        if np.all(x == np.asarray([0.05, 0.05])):
            cost = pybop.GaussianLogLikelihoodKnownSigma(dataset, sigma0=0.002)
            problem = pybop.Problem(simulator, cost)
            optim = pybop.XNES(problem)
            logger = pybop.Logger(minimising=problem.minimising)
            logger.iteration = 1
            logger.extend_log(x_search=[x], x_model=[x], cost=[problem(x)])
            result = pybop.OptimisationResult(optim=optim, logger=logger, time=1.0)

            message, _ = pybop.classify_using_hessian(result)
            assert message == "The optimiser has located a maximum."


    def test_insensitive_classify_using_hessian(self, model, parameter_values):
        param_R0_a = pybop.Parameter(bounds=[0, 0.002])
        param_R0_b = pybop.Parameter(bounds=[-0.001, 0.001])
        parameter_values.update(
            {"R0_a [Ohm]": 0.001, "R0_b [Ohm]": 0},
            check_already_exists=False,
        )
        parameter_values.update(
            {"R0 [Ohm]": Parameter("R0_a [Ohm]") + Parameter("R0_b [Ohm]")},
        )

        experiment = pybamm.Experiment(
            ["Discharge at 0.5C for 2 minutes (4 seconds period)"]
        )
        solution = pybamm.Simulation(
            model, parameter_values=parameter_values, experiment=experiment
        ).solve()
        dataset = pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": solution["Voltage [V]"].data,
            }
        )

        parameter_values.update({"R0_a [Ohm]": param_R0_a, "R0_b [Ohm]": param_R0_b})
        simulator = pybop.pybamm.Simulator(
            model, parameter_values=parameter_values, protocol=dataset
        )
        cost = pybop.SumOfPower(dataset, p=1)
        problem = pybop.Problem(simulator, cost)
        x = [0.001, 0]
        optim = pybop.XNES(problem)
        logger = pybop.Logger(minimising=problem.minimising)
        logger.iteration = 1
        logger.extend_log(x_search=[x], x_model=[x], cost=[problem(x)])
        result = pybop.OptimisationResult(optim=optim, logger=logger, time=1.0)

        message, _ = pybop.classify_using_hessian(result)
        expected1 = (
            "The cost variation is too small to classify with certainty."
            " The cost is insensitive to a change of 1e-42 in R0_b [Ohm]."
        )
        expected2 = "The cost variation is smaller than the cost tolerance: 0.01."
        assert message in {expected1, expected2}

        message, _ = pybop.classify_using_hessian(result, dx=[0.0001, 0.0001])
        assert message == (
            "The optimiser has located a minimum."
            " There may be a correlation between these parameters."
        )

        message, _ = pybop.classify_using_hessian(result, cost_tolerance=1e-2)
        assert message in {expected1, expected2}

        message, _ = pybop.classify_using_hessian(result, dx=[1, 1])
        assert message == (
            "Classification cannot proceed due to infinite cost value(s)."
            " The result is near the upper bound of R0_a [Ohm]."
        )

    def test_quadratic_hessian_full_coverage(self):
        """
        Construct a tiny deterministic quadratic problem so classify_using_hessian
        executes the full finite-difference Hessian calculation, eigen decomposition,
        the grid evaluation loop (no exceptions), and the final packing/return.
        """

        # Minimal test parameters object used by classify_using_hessian
        class TestParameters:
            def __init__(self):
                self.names = ["p0", "p1"]

            def get_bounds(self):
                return {"lower": np.array([-10.0, -10.0]), "upper": np.array([10.0, 10.0])}

        class TestProblem:
            def __init__(self, A, b, c=0.0):
                self.parameters = TestParameters()
                self.A = np.asarray(A, dtype=float)
                self.b = np.asarray(b, dtype=float)
                self.c = float(c)

            def evaluate(self, x):
                x = np.asarray(x, dtype=float)
                val = 0.5 * float(x.T @ self.A @ x) + float(self.b.dot(x)) + self.c
                return type("R", (), {"values": val})

        class TestOptim:
            def __init__(self, problem):
                self.problem = problem

        class TestResult:
            pass

        A = np.array([[4.0, 0.5], [0.5, 2.0]])
        b = np.array([0.1, -0.2])
        problem = TestProblem(A, b)

        x = np.array([1.23, -0.45], dtype=float)
        best_cost = problem.evaluate(x).values

        optim = TestOptim(problem=problem)

        result = TestResult()
        result.x = x
        result.best_cost = best_cost
        result.optim = optim
        result.minimising = True

        dx = np.array([1e-3, 1e-3], dtype=float)

        message, info = pybop.classify_using_hessian(result, dx=dx, cost_tolerance=1e-8)

        assert isinstance(info, dict)
        H = info["hessian_fd"]
        assert H.shape == (2, 2)
        assert np.isfinite(H).all()
        assert np.allclose(H, H.T, atol=1e-8)

        evals = info["eigenvalues"]
        evecs = info["eigenvectors"]
        assert evals.shape == (2,)
        assert evecs.shape == (2, 2)
        assert np.isfinite(evals).all()
        assert np.isfinite(evecs).all()

        assert "minimum" in message.lower()

        assert "param0" in info and "param1" in info and "Z" in info
        param0 = info["param0"]
        param1 = info["param1"]
        Z = info["Z"]
        assert Z.shape == (41, 41)
        assert np.isfinite(Z).all()
        assert param0.min() < x[0] < param0.max()
        assert param1.min() < x[1] < param1.max()