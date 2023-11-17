import pytest
import pybop
import numpy as np


class TestCosts:
    """
    Class for tests cost functions
    """

    @pytest.mark.unit
    def test_costs(self):
        # Construct Problem
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

        dataset = [
            pybop.Dataset("Time [s]", solution["Time [s]"].data),
            pybop.Dataset("Current function [A]", solution["Current [A]"].data),
            pybop.Dataset("Voltage [V]", solution["Terminal voltage [V]"].data),
        ]

        signal = "Voltage [V]"
        problem = pybop.Problem(model, parameters, dataset, signal=signal)

        # Base Cost
        cost = pybop.BaseCost(problem)
        assert cost.problem == problem
        with pytest.raises(NotImplementedError):
            cost([0.5])
        with pytest.raises(NotImplementedError):
            cost.n_parameters()

        # Root Mean Squared Error
        cost = pybop.RootMeanSquaredError(problem)
        cost([0.5])

        assert type(cost([0.5])) == np.float64
        assert cost([0.5]) >= 0

        # Root Mean Squared Error
        cost = pybop.SumSquaredError(problem)
        cost([0.5])

        assert type(cost([0.5])) == np.float64
        assert cost([0.5]) >= 0

        # Test catch on non-matching vector lengths
        # Sum Squared Error
        cost = pybop.SumSquaredError(problem)
        with pytest.raises(ValueError):
            cost(["test-entry"])

        with pytest.raises(ValueError):
            cost.evaluateS1(["test-entry"])

        # Root Mean Squared Error
        cost = pybop.RootMeanSquaredError(problem)
        with pytest.raises(ValueError):
            cost(["test-entry"])

    def getdata(self, model, x0):
        model.parameter_set = model.pybamm_model.default_parameter_values
        model.parameter_set.update(
            {
                "Negative electrode active material volume fraction": x0[0],
            }
        )

        sim = model.predict(t_eval=np.linspace(0, 10, 100))
        return sim
