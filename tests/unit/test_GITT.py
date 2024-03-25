import numpy as np
import pybamm
import pytest

import pybop


class TestGITT:
    """
    A class to test the GITT class.
    """

    @pytest.fixture
    def model(self):
        return "Weppner & Huggins"

    @pytest.fixture
    def parameter_set(self):
        original_parameters = pybamm.ParameterValues("Xu2019")

        return pybamm.ParameterValues(
            {
                "Reference OCP [V]": 4.1821,
                "Derivative of the OCP wrt stoichiometry [V]": -1.38636,
                "Current function [A]": original_parameters["Current function [A]"],
                "Number of electrodes connected in parallel to make a cell": original_parameters[
                    "Number of electrodes connected in parallel to make a cell"
                ],
                "Electrode width [m]": original_parameters["Electrode width [m]"],
                "Electrode height [m]": original_parameters["Electrode height [m]"],
                "Positive electrode active material volume fraction": original_parameters[
                    "Positive electrode active material volume fraction"
                ],
                "Positive electrode porosity": original_parameters[
                    "Positive electrode porosity"
                ],
                "Positive particle radius [m]": original_parameters[
                    "Positive particle radius [m]"
                ],
                "Positive electrode thickness [m]": original_parameters[
                    "Positive electrode thickness [m]"
                ],
                "Positive electrode diffusivity [m2.s-1]": original_parameters[
                    "Positive electrode diffusivity [m2.s-1]"
                ],
                "Maximum concentration in positive electrode [mol.m-3]": original_parameters[
                    "Maximum concentration in positive electrode [mol.m-3]"
                ],
            }
        )

    @pytest.fixture
    def dataset(self):
        # Define model
        original_parameters = pybamm.ParameterValues("Xu2019")
        model = pybop.lithium_ion.SPM(
            parameter_set=original_parameters, options={"working electrode": "positive"}
        )

        # Generate data
        sigma = 0.005
        t_eval = np.arange(0, 150, 2)
        values = model.predict(t_eval=t_eval)
        corrupt_values = values["Voltage [V]"].data + np.random.normal(
            0, sigma, len(t_eval)
        )

        # Return dataset
        return pybop.Dataset(
            {
                "Time [s]": t_eval,
                "Current function [A]": values["Current [A]"].data,
                "Voltage [V]": corrupt_values,
            }
        )

    @pytest.mark.unit
    def test_gitt_problem(self, model, parameter_set, dataset):
        # Test incorrect model
        with pytest.raises(ValueError):
            pybop.GITT(model="bad model", parameter_set=parameter_set, dataset=dataset)

        # Construct Problem
        problem = pybop.GITT(model, parameter_set, dataset)

        # Test fixed attributes
        parameters = [
            pybop.Parameter(
                "Positive electrode diffusivity [m2.s-1]",
                prior=pybop.Gaussian(5e-14, 1e-13),
                bounds=[1e-16, 1e-11],
                # true_value=parameter_set["Positive electrode diffusivity [m2.s-1]"],
                true_value=25,
            ),
        ]

        # assert problem.parameters == parameters

        assert problem.signal == ["Voltage [V]"]
