from mimic.core.Environment_core import Environment_core
from typing import Union
import numpy as np

class Environment(Environment_core):
    def __init__(self, objective_function, penalty_functions:list = []):
        self.objective_function = objective_function
        self.penalty_functions = penalty_functions
    
    def get_fitness(self, individual)->float:
        f = self.objective_function(individual)
        return f
    
    def get_penalty(self, individual)->float:
        p = 0.
        for p_func in self.penalty_functions:
            p += p_func(individual)
        return float(p)
    
    def set_score(self, population):
        for individual in population:
            if not individual.already_eval:
                individual.fitness = self.get_fitness(individual)
                individual.penalty = self.get_penalty(individual)
                individual.score = individual.fitness + individual.penalty