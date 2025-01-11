import numpy as np
from mimic.MOEA.Environment import Environment
from mimic import Optimizer
import mimic
import sys

class Environment_MOGA(Environment):
    def __init__(self, objective_functions:list, d_share:float, fl:np.ndarray, fu:np.ndarray, penalty_functions:list = []):
        assert d_share > 0.
        assert len(objective_functions) == len(fl)
        assert len(objective_functions) == len(fu)
        super().__init__(objective_functions, penalty_functions)
        self.d_share = d_share
        self.fl = fl
        self.fu = fu
    
    def evaluate(self, population):
        front_rank = mimic.utils.Population.get_frontRank(population)
        print(front_rank)
        sys.exit()
        # for individual in population:
        #     individual.score = individual.fitness[self.idx] + individual.penalty