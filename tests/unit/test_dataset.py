import sys

import numpy as np
import pybamm
import pytest
from scipy.integrate import cumulative_trapezoid, trapezoid

import pybop


class TestDataset:
    """
    Class to test dataset construction.
    """

    pytestmark = pytest.mark.unit

    def test_dataset(self):
        # Construct and simulate model
        model = pybamm.lithium_ion.SPM()
        solution = pybamm.Simulation(model).solve(t_eval=np.linspace(0, 10, 100))

        # Form dataset
        data_dictionary = {
            "Time [s]": solution["Time [s]"].data,
            "Current [A]": solution["Current [A]"].data,
            "Discharge capacity [A.h]": solution["Discharge capacity [A.h]"].data,
            "Voltage [V]": solution["Voltage [V]"].data,
        }
        dataset = pybop.Dataset(data_dictionary)

        # Test repr
        print(dataset)

        # Test data structure
        assert dataset.data == data_dictionary
        assert np.all(dataset["Time [s]"] == solution["Time [s]"].data)

        # Test exception for non-dictionary inputs
        with pytest.raises(
            TypeError, match="The input to pybop.Dataset must be a dictionary."
        ):
            pybop.Dataset(["StringInputShouldNotWork"])
        with pytest.raises(
            TypeError, match="The input to pybop.Dataset must be a dictionary."
        ):
            pybop.Dataset(solution["Time [s]"].data)

        # Test conversion of pybamm solution into dictionary
        assert dataset.data == pybop.import_pybamm_solution(solution).data

        # Test set and get item
        test_current = solution["Current [A]"].data + np.ones_like(
            solution["Current [A]"].data
        )
        dataset["Current [A]"] = test_current
        assert np.all(dataset["Current [A]"] == test_current)
        with pytest.raises(ValueError):
            dataset["Time"]

        # Test conversion of single signal to list
        assert dataset.check()

        # Test get subset
        dataset = dataset.get_subset(list(range(5)))
        assert len(dataset) == 5

        # Form frequency dataset
        data_dictionary = {
            "Frequency [Hz]": np.linspace(-10, 0, 10),
            "Current [A]": np.zeros(10),
            "Impedance": np.zeros(10),
        }
        frequency_dataset = pybop.Dataset(data_dictionary)

        with pytest.raises(ValueError, match="Frequencies cannot be negative."):
            frequency_dataset.check(domain="Frequency [Hz]", signal="Impedance")

    def test_pybamm_import(self):
        model = pybamm.lithium_ion.SPM()
        solution = pybamm.Simulation(model=model).solve(t_eval=np.linspace(0, 10, 100))

        # Dataset constructed from pybamm solution
        dataset_pybamm = pybop.import_pybamm_solution(
            solution, variables=["Time [s]", "Current [A]", "Voltage [V]"]
        )

        # Manually create data dictionary
        data_dictionary = {
            "Time [s]": solution["Time [s]"].data,
            "Current [A]": solution["Current [A]"].data,
            "Voltage [V]": solution["Voltage [V]"].data,
        }

        dataset_dictionary = pybop.Dataset(data_dictionary)

        assert dataset_dictionary.data == dataset_pybamm.data

    @pytest.mark.skipif(
        sys.version_info < (3, 11), reason="requires a python version >= 3.11"
    )
    @pytest.mark.skipif(
        sys.version_info >= (3, 13), reason="requires a python version < 3.13"
    )
    def test_pyprobe_import(self):
        """
        Function to test import of PyProBE Result into a pybop.Dataset.
        PyProBE requires Python 3.11 or higher.
        """
        import pyprobe

        # The example used for testing is taken from https://github.com/pybamm-team/PyBaMM/blob/develop/examples/scripts/cycling_ageing.py
        # Chosen because the experiment it includes multiple cycles and steps
        model = pybamm.lithium_ion.SPM(
            {
                "SEI": "ec reaction limited",
                "SEI film resistance": "distributed",
                "SEI porosity change": "true",
                "lithium plating": "irreversible",
                "lithium plating porosity change": "true",
            }
        )

        param = pybamm.ParameterValues("Mohtat2020")

        experiment = pybamm.Experiment(
            [
                (
                    "Charge at 1 C until 4.2 V",
                    "Hold at 4.2 V until C/10",
                    "Rest for 5 minutes",
                    "Discharge at 1 C until 2.8 V",
                    "Rest for 5 minutes",
                )
            ]
            * 2
            + [
                (
                    "Charge at 1 C until 4.2 V",
                    "Hold at 4.2 V until C/20",
                    "Rest for 30 minutes",
                    "Discharge at C/3 until 2.8 V",
                    "Rest for 30 minutes",
                ),
                (
                    "Charge at 1 C until 4.2 V",
                    "Hold at 4.2 V until C/20",
                    "Rest for 30 minutes",
                    "Discharge at 1 C until 2.8 V",
                    "Rest for 30 minutes",
                ),
                (
                    "Charge at 1 C until 4.2 V",
                    "Hold at 4.2 V until C/20",
                    "Rest for 30 minutes",
                    "Discharge at 2 C until 2.8 V",
                    "Rest for 30 minutes",
                ),
                (
                    "Charge at 1 C until 4.2 V",
                    "Hold at 4.2 V until C/20",
                    "Rest for 30 minutes",
                    "Discharge at 3 C until 2.8 V",
                    "Rest for 30 minutes",
                ),
            ]
        )

        # Create data
        sim = pybamm.Simulation(model, experiment=experiment, parameter_values=param)
        solution = sim.solve()

        # Import pybamm solution data into PyProBE
        cell = pyprobe.Cell(info={"Model": "US06"})
        cell.import_pybamm_solution("US06 DFN", ["US06"], solution)

        # Import data from PyProBE into pybop.dataset
        dataset_pyprobe = pybop.import_pyprobe_result(
            cell.procedure["US06 DFN"],
            variables=[
                "Time [s]",
                "Current [A]",
                "Voltage [V]",
                "Step",
                "Cycle",
                "Discharge capacity [A.h]",
            ],
            column_names=[
                "Time [s]",
                "Current [A]",
                "Voltage [V]",
                "Step",
                "Cycle",
                "Capacity [Ah]",
            ],
        )

        # Different choice of parameters where column names are identical between pyprobe and pybamm
        dataset_pyprobe2 = pybop.import_pyprobe_result(
            cell.procedure["US06 DFN"],
            [
                "Time [s]",
                "Current [A]",
                "Voltage [V]",
            ],
        )

        # For comparison, import a pybamm.Solution directly into pybop.Dataset
        dataset_pybamm = pybop.import_pybamm_solution(
            solution,
            variables=[
                "Time [s]",
                "Current [A]",
                "Voltage [V]",
                "Discharge capacity [A.h]",
            ],
        )

        # Ensure data is identical
        assert (dataset_pybamm["Time [s]"] == dataset_pyprobe["Time [s]"]).all()
        assert (dataset_pybamm["Current [A]"] == dataset_pyprobe["Current [A]"]).all()
        assert (dataset_pybamm["Voltage [V]"] == dataset_pyprobe["Voltage [V]"]).all()
        assert (dataset_pybamm["Step"] == dataset_pyprobe["Step"]).all()
        assert (dataset_pybamm["Cycle"] == dataset_pyprobe["Cycle"]).all()
        assert (
            dataset_pybamm["Discharge capacity [A.h]"]
            == dataset_pyprobe["Discharge capacity [A.h]"]
        ).all()
        assert (dataset_pybamm["Time [s]"] == dataset_pyprobe2["Time [s]"]).all()
        assert (dataset_pybamm["Current [A]"] == dataset_pyprobe2["Current [A]"]).all()
        assert (dataset_pybamm["Voltage [V]"] == dataset_pyprobe2["Voltage [V]"]).all()


class TestProcessing:
    """
    Class to test data processing.
    """

    pytestmark = pytest.mark.unit

    @pytest.fixture
    def dataset(self):
        # Randomly generated test dataset
        return pybop.Dataset(
            {
                "Time [s]": np.cumsum(
                    np.concatenate(([0, 1, 2], 2 * np.random.rand(25)))
                ),
                "Current [A]": np.concatenate(
                    ([0, 0, 2], np.random.normal(0, 1, 20), np.repeat(1, 5))
                ),
                "Discharge capacity [A.h]": np.cumsum(
                    np.concatenate(([0, 0, 1], np.random.normal(0, 1, 25)))
                )
                / 3600,
            }
        )

    def test_current_data_processing(self, dataset):
        # Test generation of a current consistent with the charge throughput data
        consistent_dataset = pybop.generate_consistent_current(dataset, tolerance=1e-2)
        assert len(consistent_dataset) >= len(dataset)

        for var in ["Time [s]", "Current [A]", "Discharge capacity [A.h]"]:
            assert consistent_dataset[var][0] == dataset[var][0]
            assert consistent_dataset[var][-1] == dataset[var][-1]

        charge_throughput = cumulative_trapezoid(
            y=consistent_dataset["Current [A]"],
            x=consistent_dataset["Time [s]"],
        )
        assert np.allclose(
            consistent_dataset["Discharge capacity [A.h]"][1:] * 3600, charge_throughput
        )

        # Test downsampling of constant current sections
        downsampled_dataset = pybop.downsample_constant_current(dataset, tolerance=1e-4)
        assert len(downsampled_dataset) < len(dataset)

        for var in ["Time [s]", "Current [A]", "Discharge capacity [A.h]"]:
            assert downsampled_dataset[var][0] == dataset[var][0]
            assert downsampled_dataset[var][-1] == dataset[var][-1]

        # Also check the charge throughput after making the current consistent and downsampling
        consistent_dataset = pybop.generate_consistent_current(dataset, tolerance=1e-2)
        downsampled_dataset = pybop.downsample_constant_current(consistent_dataset)

        charge_throughput = cumulative_trapezoid(
            y=downsampled_dataset["Current [A]"],
            x=downsampled_dataset["Time [s]"],
        )
        assert np.allclose(
            downsampled_dataset["Discharge capacity [A.h]"][1:] * 3600,
            charge_throughput,
        )

        # Test downsampling of constant current sections without charge throughput data
        dataset_wo_ct = pybop.Dataset(
            {
                "Time [s]": dataset["Time [s]"],
                "Current [A]": dataset["Current [A]"],
            }
        )
        ds_dataset = pybop.downsample_constant_current(dataset_wo_ct, tolerance=1e-4)
        assert len(ds_dataset) < len(dataset)

        for var in ["Time [s]", "Current [A]"]:
            assert ds_dataset[var][0] == dataset_wo_ct[var][0]
            assert ds_dataset[var][-1] == dataset_wo_ct[var][-1]

        assert np.allclose(
            trapezoid(y=ds_dataset["Current [A]"], x=ds_dataset["Time [s]"]),
            trapezoid(y=dataset_wo_ct["Current [A]"], x=dataset_wo_ct["Time [s]"]),
        )
