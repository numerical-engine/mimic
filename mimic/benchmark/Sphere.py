import numpy as np
from mimic.core.Function_core import Function

class sphere(Function):
    def forward(self, individual)->float:
        return np.sum(individual.x**2)