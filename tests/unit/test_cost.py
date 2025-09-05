import numpy as np
import pybamm
import pytest

import pybop


class TestCosts:
    """
    Class for tests cost functions
    """

    pytestmark = pytest.mark.unit

    @pytest.fixture
    def model(self):
        return pybamm.lithium_ion.SPM()

    @pytest.fixture
    def dataset(self):
        return pybop.Dataset(
            {
                "Time [s]": np.linspace(0, 360, 10),
                "Current function [A]": 1e-2 * np.ones(10),
                "Voltage [V]": np.ones(10),
            }
        )

    @pytest.fixture
    def one_parameter(self):
        return pybop.Parameter(
            "Negative electrode active material volume fraction",
            initial_value=0.6,
            bounds=[0.375, 0.625],
        )

    @pytest.fixture
    def two_parameters(self):
        return [
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Gaussian(0.6, 0.02),
                bounds=[0.58, 0.62],
            ),
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                prior=pybop.Gaussian(0.5, 0.05),
                bounds=[0.48, 0.52],
            ),
        ]

    @pytest.fixture
    def two_parameters_no_bounds(self):
        return [
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                initial_value=0.6,
            ),
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                initial_value=0.5,
            ),
        ]

    def test_pybamm_custom_cost(self, model, dataset, one_parameter):
        builder = pybop.Pybamm()
        builder.set_simulation(model)
        builder.set_dataset(dataset)
        builder.add_parameter(one_parameter)

        # Create a custom cost
        data = pybamm.DiscreteTimeData(
            dataset["Time [s]"], dataset["Voltage [V]"], "my_data"
        )
        custom_cost = pybop.costs.pybamm.custom(
            "MySumSquaredError",
            pybamm.DiscreteTimeSum((model.variables["Voltage [V]"] - data) ** 2),
            {},
        )
        builder.add_cost(custom_cost)
        problem_custom = builder.build()
        problem_custom.set_params(np.array([0.55]))

        builder = pybop.Pybamm()
        builder.set_simulation(model)
        builder.set_dataset(dataset)
        builder.add_parameter(one_parameter)
        builder.add_cost(
            pybop.costs.pybamm.SumSquaredError("Voltage [V]", "Voltage [V]")
        )
        problem = builder.build()
        problem.set_params(np.array([0.55]))

        assert problem_custom.run() == problem.run()

    @pytest.mark.parametrize(
        "pybamm_costs",
        [
            pybop.costs.pybamm.MeanSquaredError,
            pybop.costs.pybamm.RootMeanSquaredError,
            pybop.costs.pybamm.MeanAbsoluteError,
            pybop.costs.pybamm.Minkowski,
            pybop.costs.pybamm.SumOfPower,
        ],
    )
    def test_pybamm_costs(self, pybamm_costs, model, dataset, one_parameter):
        builder = pybop.Pybamm()
        builder.set_simulation(model)
        builder.set_dataset(dataset)
        builder.add_parameter(one_parameter)
        builder.add_cost(pybamm_costs("Voltage [V]", "Voltage [V]"))
        problem = builder.build()

        problem.set_params(np.array([0.55]))
        higher_cost = problem.run()
        problem.set_params(np.array([0.52]))
        lower_cost = problem.run()

        assert higher_cost > lower_cost

    @pytest.mark.parametrize(
        "pybamm_costs",
        [
            pybop.costs.pybamm.GravimetricEnergyDensity,
            pybop.costs.pybamm.GravimetricPowerDensity,
            pybop.costs.pybamm.VolumetricEnergyDensity,
            pybop.costs.pybamm.VolumetricPowerDensity,
        ],
    )
    def test_pybamm_design_costs(self, pybamm_costs, model, dataset, one_parameter):
        parameter_values = pybamm.ParameterValues("Chen2020")
        parameter_values.update(
            {
                "Electrolyte density [kg.m-3]": pybamm.Parameter(
                    "Separator density [kg.m-3]"
                ),
                "Negative electrode active material density [kg.m-3]": pybamm.Parameter(
                    "Negative electrode density [kg.m-3]"
                ),
                "Negative electrode carbon-binder density [kg.m-3]": pybamm.Parameter(
                    "Negative electrode density [kg.m-3]"
                ),
                "Positive electrode active material density [kg.m-3]": pybamm.Parameter(
                    "Positive electrode density [kg.m-3]"
                ),
                "Positive electrode carbon-binder density [kg.m-3]": pybamm.Parameter(
                    "Positive electrode density [kg.m-3]"
                ),
            },
            check_already_exists=False,
        )
        experiment = pybamm.Experiment(
            ["Discharge at 1C for 2 minutes (10 second period)"]
        )
        builder = pybop.Pybamm()
        builder.set_simulation(
            model,
            parameter_values=parameter_values,
            experiment=experiment,
        )
        builder.add_parameter(one_parameter)
        builder.add_cost(pybamm_costs())
        problem = builder.build()

        problem.set_params(np.array([0.55]))
        lower_cost = problem.run()
        problem.set_params(np.array([0.52]))
        higher_cost = problem.run()

        assert higher_cost > lower_cost  # Optimising negative cost

    @pytest.mark.parametrize(
        "pybamm_costs",
        [
            pybop.costs.pybamm.NegativeGaussianLogLikelihood,
        ],
    )
    def test_pybamm_costs_with_sigma(self, pybamm_costs, model, dataset, one_parameter):
        builder = pybop.Pybamm()
        builder.set_simulation(model)
        builder.set_dataset(dataset)
        builder.add_parameter(one_parameter)
        builder.add_cost(pybamm_costs("Voltage [V]", "Voltage [V]"))
        problem = builder.build()

        problem.set_params(np.array([0.55, 0.01]))
        higher_cost = problem.run()
        problem.set_params(np.array([0.52, 0.01]))
        lower_cost = problem.run()

        assert higher_cost > lower_cost

    def test_pybamm_scaled_cost(self, model, dataset, one_parameter):
        builder = pybop.Pybamm()
        builder.set_simulation(model)
        builder.set_dataset(dataset)
        builder.add_parameter(one_parameter)
        cost = pybop.costs.pybamm.SumOfPower("Voltage [V]", "Voltage [V]")
        builder.add_cost(pybop.costs.pybamm.ScaledCost(cost))
        problem = builder.build()

        problem.set_params(np.array([0.55]))
        higher_cost = problem.run()
        problem.set_params(np.array([0.52]))
        lower_cost = problem.run()

        assert higher_cost > lower_cost

    def test_multi_cost_weighting(self, model, dataset, one_parameter):
        def problem(weights):
            builder = pybop.builders.Pybamm()
            builder.set_simulation(model)
            builder.set_dataset(dataset)
            builder.add_parameter(one_parameter)
            builder.add_cost(
                pybop.costs.pybamm.RootMeanSquaredError("Voltage [V]", "Voltage [V]"),
                weight=weights[0],
            )
            builder.add_cost(
                pybop.costs.pybamm.RootMeanSquaredError("Voltage [V]", "Voltage [V]"),
                weight=weights[1],
            )
            return builder.build()

        problem1 = problem([1, 1])
        problem2 = problem([1, 10])

        problem2.set_params(np.array([0.55]))
        val1 = problem1.run()
        problem2.set_params(np.array([0.55]))
        val2 = problem2.run()

        np.testing.assert_allclose(val2 / val1, 5.5)

    @pytest.mark.parametrize(
        "pybop_cost",
        [
            pybop.MeanSquaredError,
            pybop.RootMeanSquaredError,
            pybop.MeanAbsoluteError,
            pybop.SumSquaredError,
            pybop.Minkowski,
            pybop.SumOfPower,
            # pybop.NegativeGaussianLogLikelihood,
        ],
    )
    def test_error_measures(self, pybop_cost):
        res = np.asarray([i for i in range(10)])
        dy = np.ones((2, 10))

        # Test w/o sensitivities
        cost = pybop_cost(weighting=1)
        val = cost(res)
        assert val > 0

        # Test w/ sensitivities
        val, grad = cost(res, dy)
        assert val > 0
        assert grad > 0

        # Test cost direction
        res1 = np.asarray([i for i in range(10)]) * 2
        res2 = np.asarray([i for i in range(10)]) * 0.5
        higher_cost = cost(res1)
        lower_cost = cost(res2)
        assert higher_cost > lower_cost

    def test_minkowski(self):
        # Incorrect order
        with pytest.raises(ValueError, match="The order of the Minkowski distance"):
            pybop.costs.pybamm.Minkowski("Voltage [V]", "Voltage [V]", p=-1)
        with pytest.raises(
            ValueError,
            match="For p = infinity, an implementation of the Chebyshev distance is required.",
        ):
            pybop.costs.pybamm.Minkowski("Voltage [V]", "Voltage [V]", p=np.inf)

    def test_sumofpower(self):
        # Incorrect order
        with pytest.raises(
            ValueError,
            match="The order of the Minkowski distance must be greater than 0.",
        ):
            pybop.costs.pybamm.Minkowski("Voltage [V]", "Voltage [V]", p=-1)

        with pytest.raises(
            ValueError,
            match="For p = infinity, an implementation of the Chebyshev distance is required.",
        ):
            pybop.costs.pybamm.Minkowski("Voltage [V]", "Voltage [V]", p=np.inf)
