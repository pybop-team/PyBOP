import numbers

import numpy as np
import pybamm
import pytest
from pybamm import IDAKLUSolver

import pybop


class TestBuilder:
    """
    A class to test the problem class.
    """

    pytestmark = pytest.mark.unit

    @pytest.fixture
    def parameter_values(self):
        return pybamm.ParameterValues("Chen2020")

    @pytest.fixture
    def dataset(self, model, parameter_values):
        solver = IDAKLUSolver(atol=1e-6, rtol=1e-6)
        sim = pybamm.Simulation(
            model(), parameter_values=parameter_values, solver=solver
        )
        sol = sim.solve(t_eval=np.linspace(0, 10, 20))
        return pybop.Dataset(
            {
                "Time [s]": sol["Time [s]"].data,
                "Current function [A]": sol["Current [A]"].data,
                "Voltage [V]": sol["Terminal voltage [V]"].data,
            }
        )

    @pytest.fixture
    def eis_dataset(self):
        return pybop.Dataset(
            {
                "Frequency [Hz]": np.logspace(-4.5, 5, 30),
                "Current function [A]": np.ones(30) * 0.0,
                "Impedance": np.ones(30) * 0.0,
            }
        )

    @pytest.fixture(
        params=[
            pybamm.lithium_ion.SPM,
            pybamm.lithium_ion.SPMe,
            pybamm.lithium_ion.DFN,
            # pybamm.lithium_ion.MPM(),
            # pybamm.lithium_ion.MSMR(options={"number of MSMR reactions": ("6", "4")}),
            # pybamm.equivalent_circuit.Thevenin(),
            # pybop.lithium_ion.WeppnerHuggins(),
            # pybop.lithium_ion.GroupedSPMe(),
        ]
    )
    def model(self, request):
        return request.param

    def test_builder(self, model, parameter_values, dataset):
        builder = pybop.builders.Pybamm()
        builder.set_dataset(dataset)
        builder.set_simulation(
            model(),
            parameter_values=parameter_values,
            solver=IDAKLUSolver(atol=1e-6, rtol=1e-6),
        )
        builder.add_parameter(
            pybop.Parameter(
                "Negative electrode active material volume fraction", initial_value=0.6
            )
        )
        builder.add_parameter(
            pybop.Parameter(
                "Positive electrode active material volume fraction", initial_value=0.6
            )
        )
        builder.add_cost(
            pybop.costs.pybamm.SumSquaredError("Voltage [V]", "Voltage [V]")
        )
        builder.add_cost(
            pybop.costs.pybamm.MeanAbsoluteError("Voltage [V]", "Voltage [V]")
        )
        problem = builder.build()

        assert problem is not None
        problem.set_params(np.array([0.6, 0.6]))
        value1 = problem.run()
        problem.set_params(np.array([0.7, 0.7]))
        value2 = problem.run()
        assert (value1 - value2) / value1 > 1e-5
        problem.set_params(np.array([0.6, 0.6]))
        value1s, grad1s = problem.run_with_sensitivities()
        assert grad1s.shape == (2,)
        problem.set_params(np.array([0.7, 0.7]))
        value2s, grad2s = problem.run_with_sensitivities()
        np.testing.assert_allclose(value1s, value1, atol=1e-5)
        np.testing.assert_allclose(value2s, value2, atol=1e-5)

        # Test building twice
        problem2 = builder.build()
        assert problem2 != problem

    def test_builder_likelihoods(self, model, parameter_values, dataset):
        builder = pybop.builders.Pybamm()
        builder.set_dataset(dataset)
        builder.set_simulation(
            model(),
            parameter_values=parameter_values,
            solver=IDAKLUSolver(atol=1e-7, rtol=1e-7),
        )
        builder.add_parameter(
            pybop.Parameter(
                "Negative electrode active material volume fraction", initial_value=0.6
            )
        )
        builder.add_parameter(
            pybop.Parameter(
                "Positive electrode active material volume fraction", initial_value=0.6
            )
        )
        builder.add_cost(
            pybop.costs.pybamm.NegativeGaussianLogLikelihood(
                "Voltage [V]", "Voltage [V]", 1e-2
            )
        )
        problem = builder.build()

        problem.set_params(np.array([0.6, 0.6]))
        value1 = problem.run()
        assert isinstance(value1, numbers.Number)
        problem.set_params(np.array([0.7, 0.7]))
        value2 = problem.run()
        assert (value1 - value2) / value1 < 0.0
        problem.set_params(np.array([0.6, 0.6]))
        value1s, grad1s = problem.run_with_sensitivities()
        assert isinstance(value1, numbers.Number)
        assert grad1s.shape == (2,)
        problem.set_params(np.array([0.7, 0.7]))
        value2s, grad2s = problem.run_with_sensitivities()
        np.testing.assert_allclose(value1s, value1, rtol=5e-5)
        np.testing.assert_allclose(value2s, value2, rtol=5e-5)

        # Test with estimated sigma
        builder.add_cost(
            pybop.costs.pybamm.NegativeGaussianLogLikelihood(
                "Voltage [V]",
                "Voltage [V]",
            )
        )
        problem2 = builder.build()
        problem2.set_params(np.array([0.6, 0.6, 1e-2]))
        value3 = problem2.run()
        np.testing.assert_allclose(2 * value1, value3)

        # Different sigma
        problem2.set_params(np.array([0.6, 0.6, 1e-3]))
        value4 = problem2.run()
        assert np.not_equal(2 * value1, value4)

    def test_builder_posterior(self, model, parameter_values, dataset):
        builder = pybop.builders.Pybamm()
        builder.set_dataset(dataset)
        builder.set_simulation(
            model(),
            parameter_values=parameter_values,
            solver=IDAKLUSolver(atol=1e-6, rtol=1e-6),
        )
        builder.add_parameter(
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                initial_value=0.6,
                prior=pybop.Gaussian(0.6, 0.1),
            )
        )
        builder.add_parameter(
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                initial_value=0.6,
                prior=pybop.Gaussian(0.6, 0.1),
            )
        )
        builder.add_cost(
            pybop.costs.pybamm.NegativeGaussianLogLikelihood(
                "Voltage [V]", "Voltage [V]", 1e-2
            )
        )
        problem = builder.build()

        assert problem._use_posterior is True
        problem.set_params(np.array([0.6, 0.6]))
        value1 = problem.run()
        problem.set_params(np.array([0.7, 0.7]))
        value2 = problem.run()
        assert (value1 - value2) / value1 < 0.0
        problem.set_params(np.array([0.6, 0.6]))
        value1s, grad1s = problem.run_with_sensitivities()
        assert grad1s.shape == (2,)
        problem.set_params(np.array([0.7, 0.7]))
        value2s, grad2s = problem.run_with_sensitivities()
        np.testing.assert_allclose(value1s, value1, rtol=1e-5)
        np.testing.assert_allclose(value2s, value2, rtol=1e-5)

    def test_builder_with_rebuild_params(self, model, parameter_values, dataset):
        builder = pybop.builders.Pybamm()
        builder.set_dataset(dataset)
        builder.set_simulation(
            model(),
            parameter_values=parameter_values,
        )
        builder.add_parameter(
            pybop.Parameter("Negative electrode thickness [m]", initial_value=1e-6)
        )
        builder.add_parameter(
            pybop.Parameter("Positive particle radius [m]", initial_value=1e-5)
        )
        builder.add_cost(
            pybop.costs.pybamm.SumSquaredError("Voltage [V]", "Voltage [V]")
        )
        builder.add_cost(
            pybop.costs.pybamm.MeanAbsoluteError("Voltage [V]", "Voltage [V]")
        )
        problem = builder.build()

        problem.set_params(np.array([1e-5, 0.5e-6]))
        value1 = problem.run()
        problem.set_params(np.array([2e-5, 1.5e-6]))
        value2 = problem.run()
        assert (value1 - value2) / value1 > 1e-5

    def test_builder_with_cost_hypers(self, model, parameter_values, dataset):
        builder = pybop.builders.Pybamm()
        builder.set_dataset(dataset)
        builder.set_simulation(
            model(),
            parameter_values=parameter_values,
            solver=IDAKLUSolver(atol=1e-6, rtol=1e-6),
        )
        builder.add_parameter(
            pybop.Parameter(
                "Negative electrode active material volume fraction", initial_value=0.6
            )
        )
        builder.add_parameter(
            pybop.Parameter(
                "Positive electrode active material volume fraction", initial_value=0.6
            )
        )

        # Add cost without a sigma parameter
        builder.add_cost(
            pybop.costs.pybamm.SumSquaredError("Voltage [V]", "Voltage [V]")
        )
        problem = builder.build()

        problem.set_params(np.array([0.6, 0.6]))
        value1 = problem.run()
        problem.set_params(np.array([0.7, 0.7]))
        value2 = problem.run()
        assert (value1 - value2) / value1 > 1e-5
        problem.set_params(np.array([0.6, 0.6]))
        value1s, grad1s = problem.run_with_sensitivities()
        assert grad1s.shape == (2,)
        problem.set_params(np.array([0.7, 0.7]))
        value2s, grad2s = problem.run_with_sensitivities()
        np.testing.assert_allclose(value1s, value1, rtol=1e-4)
        np.testing.assert_allclose(value2s, value2, rtol=1e-4)

    def test_eis_builder(self, model, parameter_values, eis_dataset):
        builder = pybop.builders.PybammEIS()
        builder.set_dataset(eis_dataset)
        builder.set_simulation(
            model(options={"surface form": "differential"}),
            parameter_values=parameter_values,
        )
        builder.add_parameter(
            pybop.Parameter(
                "Negative electrode active material volume fraction", initial_value=0.6
            )
        )
        builder.add_parameter(
            pybop.Parameter(
                "Positive electrode active material volume fraction", initial_value=0.6
            )
        )
        builder.add_cost(pybop.MeanSquaredError(weighting="equal"))
        problem = builder.build()

        problem.set_params(np.array([0.65, 0.65]))
        value1 = problem.run()
        problem.set_params(np.array([0.75, 0.75]))
        value2 = problem.run()
        assert (value1 - value2) / value1 > 1e-5

    def test_eis_builder_with_rebuild_parameters(
        self, model, parameter_values, eis_dataset
    ):
        builder = pybop.builders.PybammEIS()
        builder.set_dataset(eis_dataset)
        builder.set_simulation(
            model(options={"surface form": "differential"}),
            parameter_values=parameter_values,
        )
        builder.add_parameter(
            pybop.Parameter("Negative electrode thickness [m]", initial_value=1e-6)
        )
        builder.add_parameter(
            pybop.Parameter("Positive particle radius [m]", initial_value=1e-5)
        )
        builder.add_cost(pybop.MeanSquaredError(weighting="domain"))
        problem = builder.build()

        problem.set_params(np.asarray([80e-6, 4.5e-6]))
        value1 = problem.run()
        problem.set_params(np.asarray([85e-6, 5.5e-6]))
        value2 = problem.run()

        # Assert direction, compared to dataset impedance values of zero
        # Direction different from non-rebuild test, due to different parameter effects.
        assert (value1 - value2) / value1 < 1e-5

    def test_pure_python_builder(self):
        dataset = pybop.Dataset(
            {"Time / s": np.linspace(0, 1, 10), "Output": np.ones(10)}
        )

        def model(x: float | list):
            output = x * dataset["Time / s"] ** 2
            sse = np.sum((output - dataset["Output"]) ** 2)
            return sse

        builder = pybop.builders.Python()
        builder.add_parameter(pybop.Parameter("x", initial_value=1))
        builder.add_func(model)
        problem = builder.build()

        problem.set_params(np.array([3.0]))
        value1 = problem.run()
        assert value1 > 0

        # Test sensitivities
        def model_with_sens(x: float | list):
            output = x * dataset["Time / s"] ** 2
            sens = 2 * x * dataset["Time / s"]
            sse = np.sum((output - dataset["Output"]) ** 2)
            sse_grad = 2 * np.sum(output - dataset["Output"]) * np.sum(sens)
            return sse, sse_grad

        builder = pybop.builders.Python()
        builder.add_parameter(pybop.Parameter("x", initial_value=1))
        builder.add_func_with_sens(model_with_sens=model_with_sens)
        problem_sens = builder.build()
        problem_sens.set_params(np.asarray([3.0]))
        val, sens = problem_sens.run_with_sensitivities()
        assert val > 0
        assert sens > 0

        # Test incorrect model
        with pytest.raises(TypeError, match="Model must be callable"):
            builder.add_func([2.0])

    def test_build_with_initial_state(self, model, parameter_values, dataset):
        builder = pybop.builders.Pybamm()
        builder.set_dataset(dataset)
        builder.set_simulation(
            model(),
            parameter_values=parameter_values,
            solver=IDAKLUSolver(atol=1e-6, rtol=1e-6),
            initial_state="4.0 V",
        )
        builder.add_parameter(
            pybop.Parameter(
                "Negative electrode active material volume fraction", initial_value=0.6
            )
        )
        builder.add_parameter(
            pybop.Parameter(
                "Positive electrode active material volume fraction", initial_value=0.6
            )
        )
        builder.add_cost(
            pybop.costs.pybamm.SumSquaredError("Voltage [V]", "Voltage [V]")
        )
        problem = builder.build()

        # First build
        problem.set_params(np.array([0.6, 0.6]))
        value1 = problem.run()
        built_model_1 = problem.pipeline.built_model.new_copy()

        # Second build w/ SOC instead of Voltage
        builder.set_simulation(
            model(),
            parameter_values=parameter_values,
            solver=IDAKLUSolver(atol=1e-6, rtol=1e-6),
            initial_state=0.5,
        )
        problem2 = builder.build()
        problem2.set_params(np.array([0.6, 0.6]))
        value2 = problem2.run()
        built_model_2 = problem2.pipeline.built_model.new_copy()

        # Assert builds are different
        assert (value1 - value2) / value1 < 1e-5  # Value2 is a worse fit
        assert built_model_1 != built_model_2

    def test_build_on_eval(self, model, parameter_values, dataset):
        builder = pybop.builders.Pybamm()
        builder.set_dataset(dataset)
        builder.set_simulation(
            model(),
            parameter_values=parameter_values,
            solver=IDAKLUSolver(),
            initial_state=0.5,
            build_on_eval=False,
        )
        builder.add_parameter(
            pybop.Parameter(
                "Negative electrode active material volume fraction", initial_value=0.6
            )
        )
        builder.add_parameter(
            pybop.Parameter(
                "Positive electrode active material volume fraction", initial_value=0.6
            )
        )
        builder.add_cost(
            pybop.costs.pybamm.SumSquaredError("Voltage [V]", "Voltage [V]")
        )
        problem = builder.build()
        assert not problem.pipeline.requires_rebuild

        # First build w/o `build_on_eval`
        problem.set_params(np.array([0.5, 0.5]))
        value1 = problem.run()
        built_model_1 = problem.pipeline.built_model.new_copy()

        # Second build w/ `build_on_eval`
        builder.set_simulation(
            model(),
            parameter_values=parameter_values,
            solver=IDAKLUSolver(),
            initial_state=0.5,
            build_on_eval=True,
        )
        problem2 = builder.build()
        problem2.set_params(np.array([0.5, 0.5]))
        value2 = problem2.run()
        built_model_2 = problem2._pipeline.built_model.new_copy()

        # Assert builds are different
        assert value1 != value2
        assert built_model_1 != built_model_2

    def test_build_no_parameters(self, dataset):
        builder = pybop.builders.Python()
        builder.add_func(lambda x: x**2)
        with pytest.raises(
            ValueError, match="No parameters have been added to the builder."
        ):
            builder.build()

    def test_set_formation_concentrations(self):
        parameter_values = pybamm.ParameterValues("Chen2020")
        pybop.builders.set_formation_concentrations(parameter_values)

        assert (
            parameter_values["Initial concentration in negative electrode [mol.m-3]"]
            == 0
        )
        assert (
            parameter_values["Initial concentration in positive electrode [mol.m-3]"]
            > 0
        )
