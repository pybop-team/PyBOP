import pybop
import pybamm


class SPM:
    """

    Composition of the SPM class in PyBaMM.

    """

    def __init__(self):
        self.pybamm_model = pybamm.lithium_ion.SPM()
        self.name = "Single Particle Model"
