import numpy as np
from mimic.core.Function_core import Function

class Ackley(Function):
    def forward(self, x:np.ndarray)->float:
        raise NotImplementedError