import numpy as np
from mimic.core.Function_core import Function

class sphere(Function):
    def forward(self, x:np.ndarray)->float:
        return np.sum(x**2)