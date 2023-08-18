import pybop


class BaseModel:
    """
    Base class for PyBOP models
    """

    def __init__(self, name="Base Model"):
        # self.pybamm_model = None
        self.name = name
        # self.parameter_set = None

    def build(self):
        """
        Build the model
        """
        pass

    def sim(self):
        """
        Simulate the model
        """
        pass
