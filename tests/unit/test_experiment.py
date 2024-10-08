import pybamm
import pytest

import pybop


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

        assert [step.to_dict() for step in pybop_experiment.steps] == [
            step.to_dict() for step in pybamm_experiment.steps
        ]

        assert pybop_experiment.cycle_lengths == pybamm_experiment.cycle_lengths

        assert str(pybop_experiment) == str(pybamm_experiment)

        assert repr(pybop_experiment) == repr(pybamm_experiment)

        assert pybop_experiment.termination == pybamm_experiment.termination
