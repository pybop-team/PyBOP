import pytest
import pybop
import numpy as np


class TestDataset:
    """
    Class to test dataset construction.
    """

    @pytest.mark.unit
    def test_dataset(self):
        # Construct and simulate model
        model = pybop.lithium_ion.SPM()
        model.parameter_set = model.pybamm_model.default_parameter_values
        solution = model.predict(t_eval=np.linspace(0, 10, 100))

        # Form dataset
        data_dictionary = {
            "Time [s]": solution["Time [s]"].data,
            "Current function [A]": solution["Current [A]"].data,
            "Voltage [V]": solution["Terminal voltage [V]"].data,
        }
        dataset = pybop.Dataset(data_dictionary)

        # Test repr
        print(dataset)

        # Test data structure
        assert dataset.data == data_dictionary

        # Test exception for non-dictionary inputs
        with pytest.raises(ValueError):
            pybop.Dataset(["StringInputShouldNotWork"])
        with pytest.raises(ValueError):
            pybop.Dataset(solution["Time [s]"].data)
