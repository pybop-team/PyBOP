import numpy as np
import pytest

import pybop
from pybop.costs.evaluation import Evaluation


class SeparableParaboloidProblem(pybop.Problem):
    """
    Simple paraboloid cost:
        f(x) = (x0 - c0)**2 + (x1 - c1)**2 + c
    """

    def __init__(self, centre: np.ndarray, c: float = 0.0):
        super().__init__(simulator=None, cost=None)
        self.parameters = pybop.Parameters(
            {
                "x0": pybop.Parameter(bounds=[-10, 10]),
                "x1": pybop.Parameter(bounds=[-10, 10]),
            }
        )
        self.c0 = centre[0]
        self.c1 = centre[1]
        self.c = float(c)

    def evaluate_batch(self, inputs, calculate_sensitivities=False):
        val = []
        for x in inputs:
            val.append((x["x0"] - self.c0) ** 2 + (x["x1"] - self.c1) ** 2 + self.c)
        return Evaluation(values=np.array(val))


@pytest.fixture(params=[np.asarray([0.0, 0.0]), np.asarray([0.05, 0.05])])
def optimisation_result(request):
    """
    Build a result where result.x is the true minimiser (the paraboloid centre).
    That ensures classify_using_hessian computes a finite Hessian and eigenvalues.
    """
    centre = np.asarray(request.param, dtype=float)
    problem = SeparableParaboloidProblem(centre=centre, c=1.0)  # small offset c
    optim = pybop.XNES(problem)

    logger = pybop.Logger(minimising=True)
    logger.iteration = 1
    logger.extend_log(
        x_search=[centre],
        x_model=[centre],
        cost=problem.evaluate(centre).values,
    )
    return pybop.OptimisationResult(optim=optim, logger=logger, time=1.0)


@pytest.mark.unit
def test_classify_paraboloid_minimum_and_grid(optimisation_result):
    result = optimisation_result

    dx = np.asarray([1e-3, 1e-3])
    steps = 3

    info = pybop.classify_using_hessian(result, dx=dx, cost_tolerance=1e-8)
    pybop.plot_hessian_eigenvectors(info, steps=steps)

    # Basic structure checks
    assert isinstance(info, dict)
    assert info["cfd_hessian"].shape == (2, 2)
    assert info["eigenvalues"].shape == (2,)
    assert info["eigenvectors"].shape == (2, 2)
    assert info["x"].shape == (2,)
    assert info["dx"].shape == (2,)
    assert isinstance(info["names"], list) and len(info["names"]) == 2
    assert isinstance(info["best_cost"], float)
    assert isinstance(info["span0"], tuple) and len(info["span0"]) == 2
    assert isinstance(info["span1"], tuple) and len(info["span1"]) == 2

    # Grid checks: a finite paraboloid evaluates to finite values everywhere
    assert info["Z"].shape == (steps, steps)
    assert np.isfinite(info["Z"]).all()

    # Hessian should be finite and symmetric (within numerical tolerance)
    H = info["cfd_hessian"]
    assert np.isfinite(H).all()
    assert np.allclose(H, H.T, atol=1e-8)

    # Eigenvalues are sorted ascending and finite
    evals = info["eigenvalues"]
    assert evals[0] <= evals[1]
    assert np.isfinite(evals).all()

    # Because the paraboloid is convex and result.x is the minimiser, expect a minimum
    assert "minimum" in info["message"].lower()
