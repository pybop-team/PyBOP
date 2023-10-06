import pybop
import pints


class MLE:
    def __init__(model, x0, method):
        self.model = model
        self.x0 = x0
        self.method = method
        self.problem = pints.SingleOutputProblem(model, x0)
