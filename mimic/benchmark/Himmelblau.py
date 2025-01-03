import numpy as np
from mimic.core.Function_core import Function

class himmelblau(Function):
    def forward(self, x:np.ndarray)->float:
        assert len(x) == 2
        xx, yy = x[0], x[1]

        return (xx**2 + yy - 11.)**2 + (xx + yy**2 - 7.)**2