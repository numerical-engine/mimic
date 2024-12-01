from mimic.core.Crossover import Crossover_meta
import numpy as np
import sys

class sbx(Crossover_meta):
    """SBX crossover

    Args:
        xl (np.ndarray): Lower limit of solution
        xu (np.ndarray): Upper limit of solution
        att_xl (dict[np.ndarray]): Lower limit of addtional variables. All keys should be in individual.att_keys.
        att_xu (dict[np.ndarray]): Upper limit of addtional variables. All keys should be in individual.att_keys.
        prob (float): Hyper parameter
        eta (float): Hyper parameter
    Attributes:
        parent_num (int): Number of parents for crossover.
        offspring_num (int): Number of offsprings given from one crossover.
        xl (np.ndarray): Lower limit of solution
        xu (np.ndarray): Upper limit of solution
        att_xl (dict[np.ndarray]): Lower limit of addtional variables
        att_xu (dict[np.ndarray]): Upper limit of addtional variables
        prob (float): Hyper parameter
        eta (float): Hyper parameter
    """
    def __init__(self, xl:np.ndarray, xu:np.ndarray, prob:float = 0.9, eta:float = 15., att_xl:dict[np.ndarray] = {}, att_xu:dict[np.ndarray] = {}):
        super().__init__(2, 2)
        self.xl = xl
        self.xu = xu
        self.att_xl = att_xl
        self.att_xu = att_xu
        self.prob = prob
        self.eta = eta
    
    def _run(self, pair:list, environment)->list:
        """Returns offsprings given from pair.

        Args:
            pair (list[core.Individual.Individual]): pair of individual
            environment (core.Environment.Environment): Environment
        Returns:
            list[core.Individual.Individual]: list of individual about offspring
        """
        assert len(self.att_xl) == len(pair[0].att_keys)
        for key in self.att_xl.keys():
            assert key in pair[0].att_keys
        
        x_pair = tuple(individual.x for individual in pair)
        x_offs = self.run(x_pair)
        assert len(x_offs) == self.offspring_num

        att_offs = [{} for _ in range(self.offspring_num)]
        for key in pair[0].att_keys:
            att_pair = tuple(individual.__dict__[key] for individual in pair)
            for idx, att in enumerate(self.run(att_pair, key)):
                att_offs[idx][key] = att
        
        offspring = [type(pair[0])(x, att = att) for x, att in zip(x_offs, att_offs)]
        return offspring
    
    def run(self, x_pair:tuple[np.ndarray], key:str = None)->list[np.ndarray]:
        """Generate variables for offspring

        Args:
            x_pair (tuple[np.ndarray]): tuple of variables with respect to pair
            key (str):Key name of additional variables. If None, x_pair is about solution.
        Returns:
            list[np.ndarray]: list of variables with respect to offspring.
        """
        x1, x2 = x_pair
        xupper = np.maximum(x1, x2); xlower = np.minimum(x1, x2)
        distance = xupper - xlower + 1e-10

        if key is not None:
            xl = self.att_xl[key]
            xu = self.att_xu[key]
        else:
            xl = self.xl
            xu = self.xu

        beta1 = 1. + 2.*(x1 - xl)/distance
        beta2 = 1. + 2.*(xu - x2)/distance
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

        x1_offs = np.clip(x1_offs, self.xl, self.xu)
        x2_offs = np.clip(x2_offs, self.xl, self.xu)

        return [x1_offs, x2_offs]