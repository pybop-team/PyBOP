from pybamm.models.base_model import PyBaMMBaseModel
import numpy as np

class BaseModel(PyBaMMBaseModel):
    """ 

    This is a wrapper class for the PyBaMM Model class.
    
    """

    def __init__(self):
        """

        Insert initialisation code as needed.

        """

    def update(self, k):
        """
        Updater
        """
        print(k)