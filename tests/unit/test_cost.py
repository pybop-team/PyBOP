import pytest
import pybop
import numpy as np


class TestCosts:
    """
    Class for tests cost functions
    """

    @pytest.fixture
    def model(self):
        return pybop.lithium_ion.SPM()

    @pytest.fixture
    def parameters(self):
        return [
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Gaussian(0.5, 0.01),
                bounds=[0.375, 0.625],
            ),
        ]

    @pytest.fixture
    def experiment(self):
        return pybop.Experiment(
            [
                ("Discharge at 1C for 5 minutes (5 second period)"),
            ]
        )

    @pytest.fixture
    def x0(self):
        return np.array([0.5])

    @pytest.fixture
    def dataset(self, model, experiment, x0):
        model.parameter_set = model.pybamm_model.default_parameter_values
        model.parameter_set.update(
            {
                "Negative electrode active material volume fraction": x0[0],
            }
        )
        solution = model.predict(experiment=experiment)
        return pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": solution["Terminal voltage [V]"].data,
            }
        )

    @pytest.fixture
    def signal(self):
        return "Voltage [V]"

    @pytest.mark.parametrize("cut_off", [2.5, 3.777])
    @pytest.mark.unit
    def test_costs(self, cut_off, model, parameters, dataset, signal, x0):
        # Construct Problem
        model.parameter_set.update({"Lower voltage cut-off [V]": cut_off})
        problem = pybop.FittingProblem(model, parameters, dataset, signal=signal, x0=x0)

        # Base Cost
        base_cost = pybop.BaseCost(problem)
        assert base_cost.problem == problem
        with pytest.raises(NotImplementedError):
            base_cost._evaluate([0.5])
            base_cost._evaluateS1([0.5])

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

        # Test infeasible locations
        rmse_cost.problem._model.allow_infeasible_solutions = False
        assert rmse_cost([1.1]) == np.inf

        # Test UserWarnings
        with pytest.warns(UserWarning) as record:
            rmse_cost([1.1])
            sums_cost.evaluateS1([1.1])

        assert len(record) == 2
        for i in range(len(record)):
            assert "Non-physical point encountered" in str(record[i].message)

        # Test exception for non-numeric inputs
        with pytest.raises(ValueError):
            rmse_cost(["StringInputShouldNotWork"])
        with pytest.raises(ValueError):
            sums_cost(["StringInputShouldNotWork"])
        with pytest.raises(ValueError):
            sums_cost.evaluateS1(["StringInputShouldNotWork"])

        # Test treatment of simulations that terminated early
        # by variation of the cut-off voltage.

    @pytest.mark.unit
    def test_gravimetric_energy_density_cost(
        self, model, parameters, experiment, signal
    ):
        # Construct Problem
        problem = pybop.DesignProblem(
            model, parameters, experiment, signal=signal, init_soc=0.5
        )

        # Construct Cost
        cost = pybop.GravimetricEnergyDensity(problem)
        assert cost([0.4]) <= 0  # Should be a viable design
        assert cost([0.8]) == np.inf  # Should exceed active material + porosity < 1
        assert cost([1.4]) == np.inf  # Definitely not viable
