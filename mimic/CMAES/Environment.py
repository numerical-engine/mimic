from mimic.core.Environment_core import Environment_core
import numpy as np

class Environment(Environment_core):
    def __init__(self, objective_function, penalty_functions:list = []):
        self.objective_function = objective_function
        self.penalty_functions = penalty_functions
    
    def get_fitness(self, x:np.ndarray)->float:
        f = self.objective_function(x)
        return f
    
    def get_penalty(self, x:np.ndarray)->float:
        p = 0.
        for p_func in self.penalty_functions:
            p += p_func(x)
        return float(p)
    
    def get_score(self, X:np.ndarray)->np.ndarray:
        F = []
        for x in X:
            F.append(self.get_fitness(x) + self.get_penalty(x))
        F = np.array(F)

        return F