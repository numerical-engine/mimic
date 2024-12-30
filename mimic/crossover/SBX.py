from mimic.core.Crossover_core import Crossover
import numpy as np
import sys

class sbx(Crossover):
    def __init__(self, xl:np.ndarray, xu:np.ndarray, prob:float = 0.9, eta:float = 15.):
        super().__init__(2, 2)
        self.xl = xl
        self.xu = xu
        self.prob = prob
        self.eta = eta
    
    def run(self, x_pair:tuple[np.ndarray])->list[np.ndarray]:
        x1, x2 = x_pair
        xupper = np.maximum(x1, x2); xlower = np.minimum(x1, x2)
        distance = xupper - xlower + 1e-10

        beta1 = 1. + 2.*(x1 - self.xl)/distance
        beta2 = 1. + 2.*(self.xu - x2)/distance
        alpha1 = 2. - beta1**(-(self.eta+1))
        alpha2 = 2. - beta2**(-(self.eta+1))

        def prob1(u, alpha):
            return (u*alpha)**(1./(self.eta + 1.))
        def prob2(u, alpha):
            return (1./(2. - u*alpha))**(1./(self.eta + 1.))

        
        u = np.random.rand(len(x1))
        betaq1 = np.where(u <= 1./alpha1, prob1(u, alpha1), prob2(u, alpha1))
        betaq2 = np.where(u <= 1./alpha2, prob1(u, alpha2), prob2(u, alpha2))

        mask = (np.random.rand(len(x1)) < self.prob).astype(float)
        x1_offs = x1*(1. - mask) + mask*0.5*((x1 + x2) - betaq1*(x2 - x1))
        x2_offs = x2*(1. - mask) + mask*0.5*((x1 + x2) + betaq2*(x2 - x1))

        return [x1_offs, x2_offs]