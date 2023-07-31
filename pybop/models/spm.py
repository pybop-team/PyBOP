import pybamm
from .base_model import BaseModel


class BaseSPM(pybamm.models.full_battery_models.lithium_ion.BasicSPM):
    """

    Inherites from the BasicSPM class in PyBaMM

    """

    def __init__(self):
        super().__init__()
