import pytest
import pybop
import pybamm
import numpy as np


class TestExperiment:
    """
    Class to test the experiment class.
    """

    @pytest.mark.unit
    def test_experiment(self):
        # Define example protocol
        protocol = [("Discharge at 1 C for 20 seconds")]

        # Construct matching experiments
        pybop_experiment = pybop.Experiment(protocol)
        pybamm_experiment = pybamm.Experiment(protocol)

        assert [
            step.to_dict() for step in pybop_experiment.operating_conditions_steps
        ] == [step.to_dict() for step in pybamm_experiment.operating_conditions_steps]

        assert pybop_experiment.cycle_lengths == pybamm_experiment.cycle_lengths

        assert str(pybop_experiment) == str(pybamm_experiment)

        assert repr(pybop_experiment) == repr(pybamm_experiment)

        assert pybop_experiment.termination == pybamm_experiment.termination
