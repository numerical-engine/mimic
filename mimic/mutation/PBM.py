import numpy as np
from mimic.core.Mutation_core import Mutation

class pbm(Mutation):
    def __init__(self, xl:np.ndarray, xu:np.ndarray, prob:float = 0.1, eta:float = 20.):
        super().__init__()
        self.prob = prob
        self.eta = eta
        self.xl = xl
        self.xu = xu
    
    def run(self, individual):
        delta_l = (individual.x - self.xl)/(self.xu - self.xl)
        delta_u = (self.xu - individual.x)/(self.xu - self.xl)

        u = np.random.rand(len(delta_l))
        delta = np.where(u <= 0.5,
                         (2.*u + (1.-2.*u)*(1. - delta_l)**(1./(self.eta + 1.))) - 1.,
                         1. - (2.*(1. - u) + (2.*u - 1.)*(1. - delta_u)**(self.eta + 1.))**(1./(self.eta + 1.)))

        mask = (np.random.rand(len(delta_l)) < self.prob).astype(float)
        delta *= mask*(self.xu - self.xl)

        individual.x += delta

        return individual