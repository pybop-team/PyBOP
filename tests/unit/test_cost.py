import pytest
import pybop
import numpy as np


class TestCosts:
    """
    Class for tests cost functions
    """

    @pytest.mark.parametrize("cut_off", [2.5, 3.777])
    @pytest.mark.unit
    def test_costs(self, cut_off):
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
        signal = "Voltage [V]"
        model.parameter_set.update({"Lower voltage cut-off [V]": cut_off})
        problem = pybop.FittingProblem(model, parameters, dataset, signal=signal, x0=x0)

        # Base Cost
        base_cost = pybop.BaseCost(problem)
        assert base_cost.problem == problem
        with pytest.raises(NotImplementedError):
            base_cost([0.5])

        # Root Mean Squared Error
        rmse_cost = pybop.RootMeanSquaredError(problem)
        rmse_cost([0.5])

        # Sum Squared Error
        sums_cost = pybop.SumSquaredError(problem)
        sums_cost([0.5])

        # Test type of returned value
        assert type(rmse_cost([0.5])) == np.float64
        assert rmse_cost([0.5]) >= 0

        assert type(sums_cost([0.5])) == np.float64
        assert sums_cost([0.5]) >= 0
        e, de = sums_cost.evaluateS1([0.5])

        assert type(e) == np.float64
        assert type(de) == np.ndarray

        # Test option setting
        sums_cost.set_fail_gradient(1)

        # Test exception for non-numeric inputs
        with pytest.raises(ValueError):
            rmse_cost(["StringInputShouldNotWork"])
        with pytest.raises(ValueError):
            sums_cost(["StringInputShouldNotWork"])
        with pytest.raises(ValueError):
            sums_cost.evaluateS1(["StringInputShouldNotWork"])

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
