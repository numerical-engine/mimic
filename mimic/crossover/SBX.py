from mimic.core.Crossover_core import Crossover
import numpy as np
import sys

class sbx(Crossover):
    def __init__(self, prob:float = 0.9, eta:float = 15.):
        super().__init__(2, 2)
        self.prob = prob
        self.eta = eta
    
    def run(self, x_pair:tuple[np.ndarray])->list[np.ndarray]:
        x1, x2 = x_pair
        U = np.random.rand(len(x1))
        beta = []
        for u in U:
            b = 2.*u if u <= 0.5 else 1./(2.*(1. - u))
            beta.append(b**(1./(self.eta + 1.)))
        beta = np.array(beta)

        mask = (np.random.rand(len(x1)) < self.prob).astype(float)
        x1_offs = x1*(1. - mask) + mask*0.5*((x1 + x2) - beta*(x2 - x1))
        x2_offs = x2*(1. - mask) + mask*0.5*((x1 + x2) + beta*(x2 - x1))

        return [x1_offs, x2_offs]