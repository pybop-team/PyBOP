import pybop


class BaseModel:
    """
    Base class for PyBOP models.
    """

    def __init__(self, name="Base Model"):
        self.name = name

    def build(
        self,
        dataset=None,
        parameters=None,
        check_model=True,
        init_soc=None,
    ):
        """
        Build the model (if not built already).
        """
        self.dataset = dataset
        self.parameters = parameters

        raise ValueError("Not yet implemented")

    def set_init_soc(self, init_soc):
        """
        Set the initial state of charge.
        """
        raise ValueError("Not yet implemented")

    def set_params(self):
        """
        Set each parameter in the model either equal to its value
        or mark it as an input.
        """
        raise ValueError("Not yet implemented")

    def simulate(self, inputs=None, t_eval=None, parameter_set=None, experiment=None):
        """
        Run the forward model and return the result in Numpy array format
        aligning with Pints' ForwardModel simulate method.
        """
        raise ValueError("Not yet implemented")

    def _simulate(self, parameter_set=None, experiment=None):
        """
        Return the results of a simulation.
        """
        raise ValueError("Not yet implemented")

    def n_parameters(self):
        """
        Returns the dimension of the parameter space.
        """
        return len(self.parameters)

    def n_outputs(self):
        """
        Returns the number of outputs this model has. The default is 1.
        """
        return 1

    @property
    def built_model(self):
        return self._built_model

    @property
    def parameter_set(self):
        return self._parameter_set

    @parameter_set.setter
    def parameter_set(self, parameter_set):
        self._parameter_set = parameter_set.copy()

    @property
    def model_with_set_params(self):
        return self._model_with_set_params
