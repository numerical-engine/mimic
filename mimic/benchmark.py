import numpy as np
from mimic.core.Function_core import Function

class sphere(Function):
    def forward(self, x:np.ndarray)->float:
        return np.sum(x**2)

class Rosenbrock(Function):
    def __init__(self, alpha:float):
        self.alpha = alpha
    def forward(self, x:np.ndarray)->float:
        f = 0.
        for i in range(len(x)-1):
            f += self.alpha*(x[i+1]-x[i]**2)**2 + (1.-x[i])**2
        return f

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

class himmelblau(Function):
    def forward(self, x:np.ndarray)->float:
        assert len(x) == 2
        xx, yy = x[0], x[1]

        return (xx**2 + yy - 11.)**2 + (xx + yy**2 - 7.)**2

class Beale(Function):
    def forward(self, x:np.ndarray)->float:
        return (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2