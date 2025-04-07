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
            param.update(parameter.name, default_value)
