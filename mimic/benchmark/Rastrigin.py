import numpy as np
from mimic.core.Function_core import Function

class Rastrigin(Function):
    def __init__(self, n:int):
        assert n >= 1, f"n should be greater than 1"
        self.n = n
    def forward(self, individual)->float:
        assert self.n == individual.dim()
        f = 10.*self.n
        for x in individual.x:
            f += (x**2 - 10.*np.cos(2.*np.pi*x))
        return f