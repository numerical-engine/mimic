from mimic.core.Environment_core import Environment_core
import numpy as np

class Environment(Environment_core):
    def __init__(self, objective_functions:list, penalty_functions:list = []):
        super().__init__(penalty_functions)
        self.objective_functions = objective_functions
    
    def get_fitness(self, individual)->np.ndarray:
        f = np.array([objective_function(individual.x) for objective_function in self.objective_functions])
        return f