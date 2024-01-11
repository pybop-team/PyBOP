from __future__ import annotations
import pytest
import pybop
import numpy as np


class TestCosts:
    """
    Class for tests cost functions
    """

    @pytest.fixture(params=[2.5, 3.777])
    def problem(self, request):
        cut_off = request.param
        # Construct model
        model = pybop.lithium_ion.SPM()

        parameters = [
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Gaussian(0.5, 0.02),
                bounds=[0.375, 0.625],
            )
        ]

        # Form dataset
        x0 = np.array([0.52])
        solution = self.getdata(model, x0)
        dataset = pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": solution["Terminal voltage [V]"].data,
            }
        )

        # Construct Problem
        signal = ["Voltage [V]"]
        model.parameter_set.update({"Lower voltage cut-off [V]": cut_off})
        problem = pybop.FittingProblem(model, parameters, dataset, signal=signal, x0=x0)
        return problem

    @pytest.fixture(
        params=[pybop.RootMeanSquaredError, pybop.SumSquaredError, pybop.ObserverCost]
    )
    def cost(self, problem, request):
        cls = request.param
        inputs = {p.name: problem.x0[i] for i, p in enumerate(problem.parameters)}
        if cls == pybop.RootMeanSquaredError or cls == pybop.SumSquaredError:
            return cls(problem)
        elif cls == pybop.ObserverCost:
            return cls(
                problem,
                observer=pybop.UnscentedKalmanFilterObserver(
                    problem._model,
                    inputs,
                    problem.signal,
                    sigma0=1e-4,
                    process=1e-4,
                    measure=1e-4,
                ),
            )

    @pytest.mark.unit
    def test_base(self, problem):
        base_cost = pybop.BaseCost(problem)
        assert base_cost.problem == problem
        with pytest.raises(NotImplementedError):
            base_cost._evaluate([0.5])
            base_cost._evaluateS1([0.5])

    @pytest.mark.unit
    def test_costs(self, cost):
        higher_cost = cost([0.5])
        lower_cost = cost([0.52])
        assert higher_cost > lower_cost or (
            higher_cost == lower_cost and higher_cost == np.inf
        )

        # Test type of returned value
        assert type(cost([0.5])) == np.float64

        if isinstance(cost, pybop.SumSquaredError):
            e, de = cost.evaluateS1([0.5])

            assert type(e) == np.float64
            assert type(de) == np.ndarray

            # Test option setting
            cost.set_fail_gradient(1)

            # Test exception for non-numeric inputs
            with pytest.raises(ValueError):
                cost.evaluateS1(["StringInputShouldNotWork"])

        # Test exception for non-numeric inputs
        with pytest.raises(ValueError):
            cost(["StringInputShouldNotWork"])

        # Test treatment of simulations that terminated early
        # by variation of the cut-off voltage.

    def getdata(self, model, x0):
        model.parameter_set = model.pybamm_model.default_parameter_values
        model.parameter_set.update(
            {
                "Negative electrode active material volume fraction": x0[0],
            }
        )

        sim = model.predict(t_eval=np.linspace(0, 10, 100))
        return sim
