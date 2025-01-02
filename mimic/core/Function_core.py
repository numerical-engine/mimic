import numpy as np

class Function:
    def __call__(self, x:np.ndarray)->float:
        return self.forward(x)
    def forward(self, x:np.ndarray)->float:
        raise NotImplementedError

class PenaltyFunction(Function):
    def __init__(self, weight:float = 1.):
        assert weight > 0., f"weight should be greater than zero."
        self.weight = weight
    def __call__(self, x:np.ndarray)->float:
        penalty_values = np.max([self.forward(x), 0.])
        return self.weight*penalty_values
    def forward(self, x:np.ndarray)->float:
        raise NotImplementedError

class PenaltyLower(PenaltyFunction):
    def __init__(self, xl:np.ndarray, weight:float = 1.):
        super().__init__(weight)
        self.xl = xl
    def forward(self, x:np.ndarray)->float:
        return np.sum(np.maximum(self.xl - x, 0.))

class PenaltyUpper(PenaltyFunction):
    def __init__(self, xu:np.ndarray, weight:float = 1.):
        super().__init__(weight)
        self.xu = xu
    def forward(self, x:np.ndarray)->float:
        return np.sum(np.maximum(x - self.xu, 0.))