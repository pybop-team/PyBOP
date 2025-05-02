import pybamm


class PybammCost:
    @staticmethod
    def variable_name():
        """
        Name of the corresponding cost variable added to pybamm model.
        """
        raise NotImplementedError()

    @staticmethod
    def variable_expression(model):
        """
        The cost/likelihood expression to be added to pybamm model.
        """
        raise NotImplementedError()

    @staticmethod
    def parameters() -> [(pybamm.Parameter, float)]:
        """
        Adds any parameters needed by the cost to the pybamm model
        """
        raise NotImplementedError()

    def add_to_model(self, model: pybamm.BaseModel, param: pybamm.ParameterValues):
        model.variables[self.variable_name()] = self.variable_expression(model)
        for parameter, default_value in self.parameters():
            param.update({parameter: default_value}, check_already_exists=False)


class PybammSumSquaredError(PybammCost):
    """
    A SumSquaredError cost implementation within Pybamm.
    """

    def variable_name(self):
        return "SumSquaredError"

    @staticmethod
    def variable_expression(model):
        return pybamm.ExplicitTimeIntegral(
            (pybamm.Variable("Voltage [V]") - pybamm.Variable("Data Voltage [V]")) ** 2,
            pybamm.Scalar(0.0),
        )

    @staticmethod
    def parameters():
        return []
