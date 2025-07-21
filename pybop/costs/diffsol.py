import pybop_diffsol


class DiffsolCost:
    def __init__(
        self,
        cost_type: pybop_diffsol.CostType,
        variable_name: str,
        data_name: str,
    ):
        self.data_name = data_name
        self.cost_type = cost_type
        self.variable_name = variable_name


# CostType.NegativeGaussianLogLikelihood(),
#    CostType.SumOfPower(p),
#    CostType.Minkowski(p),
#    CostType.SumOfSquares(),
#    CostType.MeanAbsoluteError(),
#    CostType.MeanSquaredError(),
#    CostType.RootMeanSquaredError(),
# ]


class DiffsolNegativeGaussianLogLikelihood(DiffsolCost):
    def __init__(self, variable_name: str, data_name: str):
        super().__init__(
            pybop_diffsol.CostType.NegativeGaussianLogLikelihood(),
            variable_name,
            data_name,
        )


class DiffsolSumOfPower(DiffsolCost):
    def __init__(self, variable_name: str, data_name: str, p: int):
        super().__init__(pybop_diffsol.CostType.SumOfPower(p), variable_name, data_name)


class DiffsolMinkowski(DiffsolCost):
    def __init__(self, variable_name: str, data_name: str, p: int):
        super().__init__(pybop_diffsol.CostType.Minkowski(p), variable_name, data_name)


class DiffsolSumOfSquares(DiffsolCost):
    def __init__(self, variable_name: str, data_name: str):
        super().__init__(
            pybop_diffsol.CostType.SumOfSquares(), variable_name, data_name
        )


class DiffsolMeanAbsoluteError(DiffsolCost):
    def __init__(self, variable_name: str, data_name: str):
        super().__init__(
            pybop_diffsol.CostType.MeanAbsoluteError(), variable_name, data_name
        )


class DiffsolMeanSquaredError(DiffsolCost):
    def __init__(self, variable_name: str, data_name: str):
        super().__init__(
            pybop_diffsol.CostType.MeanSquaredError(), variable_name, data_name
        )


class DiffsolRootMeanSquaredError(DiffsolCost):
    def __init__(self, variable_name: str, data_name: str):
        super().__init__(
            pybop_diffsol.CostType.RootMeanSquaredError(),
            variable_name,
            data_name,
        )
