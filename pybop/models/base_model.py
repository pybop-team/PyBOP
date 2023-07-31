from pybamm.models.base_model import BaseModel


class BaseModel(BaseModel):
    """

    This is a wrapper class for the PyBaMM Model class.

    """

    def __init__(self):
        """

        Insert initialisation code as needed.

        """

        self.name = "BaseModel"

    def update(self, k):
        """
        Updater
        """
        print(k)
