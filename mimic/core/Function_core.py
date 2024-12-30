import numpy as np

class Function:
    def __call__(self, individual)->float:
        return self.forward(individual)
    def forward(self, individual)->float:
        raise NotImplementedError

class PenaltyFunction(Function):
    def __init__(self, weight:float = 1.):
        assert weight > 0., f"weight should be greater than zero."
        self.weight = weight
    def __call__(self, individual)->float:
        penalty_values = np.max([self.forward(individual), 0.])
        return self.weight*penalty_values
    def forward(self, individual)->float:
        raise NotImplementedError

class PenaltyLower(PenaltyFunction):
    def __init__(self, xl:np.ndarray, weight:float = 1.):
        super().__init__(weight)
        self.xl = xl
    def forward(self, individual)->float:
        return np.sum(np.maximum(self.xl - individual.x, 0.))

class PenaltyUpper(PenaltyFunction):
    def __init__(self, xu:np.ndarray, weight:float = 1.):
        super().__init__(weight)
        self.xu = xu
    def forward(self, individual)->float:
        return np.sum(np.maximum(individual.x - self.xu, 0.))