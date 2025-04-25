import pybamm


class PybammCost:
    @staticmethod
    def variable_name():
        raise NotImplementedError()

    @staticmethod
    def variable_expression():
        raise NotImplementedError()

    @staticmethod
    def parameters() -> [(pybamm.Parameter, float)]:
        raise NotImplementedError()

    def add_to_model(self, model: pybamm.BaseModel, param: pybamm.ParameterValues):
        model.variables[self.variable_name()] = self.variable_expression()
        for parameter, default_value in self.parameters():
            param.update({parameter: default_value}, check_already_exists=False)


class PybammSumSquaredError(PybammCost):
    """
    A SumSquaredError cost implementation within Pybamm.
    """

    @staticmethod
    def variable_name():
        return "SumSquaredError"

    @staticmethod
    def variable_expression():
        return pybamm.ExplicitTimeIntegral(
            (pybamm.Variable("Voltage [V]") - pybamm.Variable("Data Voltage [V]")) ** 2,
            pybamm.Scalar(0.0),
        )

    @staticmethod
    def parameters():
        return [(None, None)]
