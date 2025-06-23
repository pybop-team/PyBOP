import numpy as np
import pybamm
import pytest
from pybamm import IDAKLUSolver

import pybop


class TestProblem:
    """
    A class to test the problem class.
    """

    pytestmark = pytest.mark.unit

    @pytest.fixture
    def first_model(self):
        return pybamm.lithium_ion.SPM()

    @pytest.fixture
    def second_model(self):
        return pybamm.lithium_ion.SPMe()

    @pytest.fixture
    def parameters(self):
        return pybop.Parameters(
            pybop.Parameter(
                "Negative particle radius [m]",
                prior=pybop.Gaussian(2e-05, 0.1e-5),
                bounds=[1e-6, 5e-5],
            ),
            pybop.Parameter(
                "Positive particle radius [m]",
                prior=pybop.Gaussian(0.5e-05, 0.1e-5),
                bounds=[1e-6, 5e-5],
            ),
        )

    @pytest.fixture
    def parameter_values(self):
        return pybamm.ParameterValues("Chen2020")

    @pytest.fixture
    def experiment(self):
        return pybamm.Experiment(
            [
                (
                    "Discharge at 1C for 5 minutes (1 second period)",
                    "Rest for 2 minutes (1 second period)",
                    "Charge at 1C for 5 minutes (1 second period)",
                    "Rest for 2 minutes (1 second period)",
                ),
            ]
            * 2
        )

    @pytest.fixture
    def dataset(self, first_model, parameter_values, experiment):
        sim = pybamm.Simulation(
            first_model, experiment, parameter_values=parameter_values
        )
        sol = sim.solve()
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
                "Frequency [Hz]": np.logspace(-4, 5, 30),
                "Current function [A]": np.ones(30) * 0.0,
                "Impedance": np.ones(30) * 0.0,
            }
        )

    @pytest.fixture
    def signal(self):
        return "Voltage [V]"

    @pytest.mark.parametrize(
        "model",
        [
            pybamm.lithium_ion.SPM(),
            pybamm.lithium_ion.SPMe(),
            pybamm.lithium_ion.DFN(),
            pybamm.lithium_ion.MPM(),
            pybamm.lithium_ion.MSMR(options={"number of MSMR reactions": ("6", "4")}),
            pybamm.equivalent_circuit.Thevenin(),
            pybop.lithium_ion.WeppnerHuggins(),
            pybop.lithium_ion.GroupedSPMe(),
        ],
    )
    def test_builder(self, parameter_values, experiment, dataset, model):
        builder = pybop.builders.Pybamm()
        builder.set_dataset(dataset)
        builder.set_simulation(
            model,
            solver=IDAKLUSolver(atol=1e-6, rtol=1e-6),
        )
        if isinstance(model, pybamm.equivalent_circuit.Thevenin):
            builder.add_parameter(pybop.Parameter("R0 [Ohm]", initial_value=1e-3))
        else:
            builder.add_parameter(
                pybop.Parameter(
                    "Positive electrode active material volume fraction",
                    initial_value=0.6,
                )
            )
        builder.add_cost(pybop.PybammSumSquaredError("Voltage [V]", "Voltage [V]", 1.0))
        problem = builder.build()

        assert problem is not None

    def test_builder_with_rebuild_params(
        self, first_model, parameter_values, experiment, dataset
    ):
        builder = pybop.builders.Pybamm()
        builder.set_dataset(dataset)
        builder.set_simulation(
            first_model,
            parameter_values=parameter_values,
        )
        builder.add_parameter(
            pybop.Parameter("Negative electrode thickness [m]", initial_value=1e-6)
        )
        builder.add_parameter(
            pybop.Parameter("Positive particle radius [m]", initial_value=1e-5)
        )
        builder.add_cost(pybop.PybammSumSquaredError("Voltage [V]", "Voltage [V]"))
        problem = builder.build()

        assert problem is not None

    def test_eis_builder(self, first_model, parameter_values, experiment, eis_dataset):
        builder = pybop.builders.PybammEIS()
        builder.set_dataset(eis_dataset)
        builder.set_simulation(first_model, parameter_values=parameter_values)
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
        builder.add_cost(pybop.NewMeanSquaredError(weighting="equal"))
        problem = builder.build()

        assert problem is not None

    def test_pure_python_builder(self):
        dataset = pybop.Dataset(
            {"Time / s": np.linspace(0, 1, 10), "Current [A]": np.ones(10)}
        )

        def model(x):
            return x**2

        builder = pybop.builders.Python()
        builder.add_func(model)
        builder.set_dataset(dataset)

        # builder.add_cost()
        # problem = builder.build()

        # Assertion to add
        # Parameters

    def test_build_with_initial_state(
        self, first_model, parameter_values, experiment, dataset
    ):
        builder = pybop.builders.Pybamm()
        builder.set_dataset(dataset)
        builder.set_simulation(
            first_model,
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
        builder.add_cost(pybop.PybammSumSquaredError("Voltage [V]", "Voltage [V]", 1.0))
        problem = builder.build()

        assert problem is not None
