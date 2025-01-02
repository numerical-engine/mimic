import numpy as np
from mimic.core.Function_core import Function

class Rastrigin(Function):
    def __init__(self, n:int):
        assert n >= 1, f"n should be greater than 1"
        self.n = n
    def forward(self, x:np.ndarray)->float:
        assert self.n == len(x)
        f = 10.*self.n
        for _x in x:
            f += (_x**2 - 10.*np.cos(2.*np.pi*_x))
        return f