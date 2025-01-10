from mimic.core.Environment_core import Environment_core
import numpy as np

class Environment(Environment_core):
    def __init__(self, objective_function, penalty_functions:list = []):
        super().__init__(penalty_functions)
        self.objective_function = objective_function
    
    def get_fitness(self, x:np.ndarray)->float:
        f = self.objective_function(x)
        return f
    
    def get_score(self, X:np.ndarray)->np.ndarray:
        F = []
        for x in X:
            F.append(self.get_fitness(x) + self.get_penalty(x))
        F = np.array(F)

        return F
    
    def set_score(self, population):
        raise NotImplementedError