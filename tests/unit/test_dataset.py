import sys

import numpy as np
import pybamm
import pytest

import pybop


class TestDataset:
    """
    Class to test dataset construction.
    """

    pytestmark = pytest.mark.unit

    def test_dataset(self):
        # Construct and simulate model
        model = pybop.lithium_ion.SPM()
        solution = model.predict(t_eval=np.linspace(0, 10, 100))

        # Form dataset
        data_dictionary = {
            "Time [s]": solution["Time [s]"].data,
            "Current [A]": solution["Current [A]"].data,
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
        assert dataset.data == pybop.Dataset(solution).data

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
        assert len(dataset[dataset.domain]) == 5

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
        dataset_pybamm = pybop.Dataset(
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
        sys.version_info < (3, 11), reason="requires python3.11 or higher"
    )
    def test_pyrobe_import(self):
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

        sim = pybamm.Simulation(model, experiment=experiment, parameter_values=param)
        solution = sim.solve()

        import pyprobe

        cell = pyprobe.Cell(info={"Model": "US06"})
        cell.import_pybamm_solution("US06 DFN", ["US06"], solution)

        dataset = pybop.import_pybrobe_result(
            cell.procedure["US06 DFN"],
            [
                "Time [s]",
                "Current [A]",
                "Voltage [V]",
                "Step",
                "Cycle",
                "Discharge capacity [A.h]",
            ],
            pyprobe_columns=[
                "Time [s]",
                "Current [A]",
                "Voltage [V]",
                "Step",
                "Cycle",
                "Capacity [Ah]",
            ],
        )

        dataset2 = pybop.Dataset(
            solution,
            variables=[
                "Time [s]",
                "Current [A]",
                "Voltage [V]",
                "Discharge capacity [A.h]",
            ],
        )

        assert (dataset2["Time [s]"] == dataset["Time [s]"]).all()
        assert (dataset2["Current [A]"] == dataset["Current [A]"]).all()
        assert (dataset2["Voltage [V]"] == dataset["Voltage [V]"]).all()
        assert (dataset2["Step"] == dataset["Step"]).all()
        assert (dataset2["Cycle"] == dataset["Cycle"]).all()
        assert (
            dataset2["Discharge capacity [A.h]"] == dataset["Discharge capacity [A.h]"]
        ).all()
