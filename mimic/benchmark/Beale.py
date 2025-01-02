import numpy as np
from mimic.core.Function_core import Function

class Beale(Function):
    def forward(self, x:np.ndarray)->float:
        return (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2