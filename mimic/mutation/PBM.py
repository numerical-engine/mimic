import numpy as np
from mimic.core.Mutation_core import Mutation
import sys

class pbm(Mutation):
    def __init__(self, xl:np.ndarray, xu:np.ndarray, prob:float = 0.1, eta:float = 20.):
        super().__init__()
        self.prob = prob
        self.eta = eta
        self.xl = xl
        self.xu = xu
    
    def run(self, individual):
        x = individual.x
        U = np.random.rand(len(x))
        delta = []
        for u in U:
            if u <= 0.5:
                delta.append((2*u)**(1./(self.eta+1)) - 1.)
            else:
                delta.append(1. - (2*(1-u))**(1./(self.eta+1)))
        delta = np.array(delta)
        mask = (np.random.rand(len(x)) < self.prob).astype(float)
        delta *= mask*(self.xu - self.xl)

        individual.x += delta
        return individual