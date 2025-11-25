import numpy as np
import pytest

from pybop.analysis.classification import classify_using_hessian


class ParaboloidParameters:
    """Parameter container: names and bounds for tests."""

    def __init__(self):
        self.names = ["R0 [Ohm]", "R1 [Ohm]"]

    def get_bounds(self):
        return {"lower": np.array([-10.0, -10.0]), "upper": np.array([10.0, 10.0])}


class SeparableParaboloidProblem:
    """
    Simple paraboloid cost:
        f(x) = (x0 - c0)**2 + (x1 - c1)**2 + c
    """

    def __init__(self, centre: np.ndarray, c: float = 0.0):
        self.parameters = ParaboloidParameters()
        self.centre = np.asarray(centre, dtype=float)
        self.c = float(c)

    def evaluate(self, inputs):
        x = np.asarray(inputs, dtype=float)
        if x.ndim > 1:
            x = x[0]
        val = float(((x - self.centre) ** 2).sum() + self.c)
        return type("Evaluation", (), {"values": val})


class OptimContainer:
    """Container for the problem attribute."""

    def __init__(self, problem):
        self.problem = problem


class ResultContainer:
    """Container for the result attribute."""

    pass


@pytest.fixture(params=[np.asarray([0.0, 0.0]), np.asarray([0.05, 0.05])])
def optimisation_result(request):
    """
    Build a result where result.x is the true minimiser (the paraboloid centre).
    That ensures classify_using_hessian computes a finite Hessian and eigenvalues.
    """
    centre = np.asarray(request.param, dtype=float)
    problem = SeparableParaboloidProblem(centre=centre, c=1.0)  # small offset c
    optim = OptimContainer(problem)

    result = ResultContainer()
    result.x = centre.copy()
    result.best_cost = float(problem.evaluate(centre).values)
    result.optim = optim
    result.minimising = True
    return result


@pytest.mark.unit
def test_classify_paraboloid_minimum_and_grid(optimisation_result):
    result = optimisation_result

    dx = np.array([1e-3, 1e-3], dtype=float)

    message, info = classify_using_hessian(result, dx=dx, cost_tolerance=1e-8)

    # Basic structure checks
    assert isinstance(info, dict)
    assert info["hessian_fd"].shape == (2, 2)
    assert info["eigenvalues"].shape == (2,)
    assert info["eigenvectors"].shape == (2, 2)
    assert info["x"].shape == (2,)
    assert info["dx"].shape == (2,)
    assert isinstance(info["names"], list) and len(info["names"]) == 2
    assert isinstance(info["best_cost"], float)
    assert isinstance(info["span0"], tuple) and len(info["span0"]) == 2
    assert isinstance(info["span1"], tuple) and len(info["span1"]) == 2

    # Grid checks: a finite paraboloid evaluates to finite values everywhere
    assert info["Z"].shape == (41, 41)
    assert np.isfinite(info["Z"]).all()

    # Hessian should be finite and symmetric (within numerical tolerance)
    H = info["hessian_fd"]
    assert np.isfinite(H).all()
    assert np.allclose(H, H.T, atol=1e-8)

    # Eigenvalues are sorted ascending and finite
    evals = info["eigenvalues"]
    assert evals[0] <= evals[1]
    assert np.isfinite(evals).all()

    # Because the paraboloid is convex and result.x is the minimiser, expect a minimum
    assert "minimum" in message.lower()
