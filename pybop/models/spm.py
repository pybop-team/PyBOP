import pybop
import pybamm
from .base_model import BaseModel

class BaseSPM():
    """

    Implements base SPM from PyBaMM

    """

    def __init__(self):
        """

        Insert initialisation code as needed.

        """

        self.name = "Base SPM"
        self.model = pybamm.lithium_ion.SPM()
        