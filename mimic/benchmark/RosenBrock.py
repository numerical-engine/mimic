import numpy as np
from mimic.core.Function_core import Function

class Rosenbrock(Function):
    def __init__(self, n:int):
        assert n >= 2., f"n should be greater than 2"
        self.n = n
    def forward(self, x:np.ndarray)->float:
        f = 0.
        for i in range(len(x)-1):
            f += 100.*(x[i+1]-x[i]**2)**2 + (1.-x[i])**2
        return f