from mimic.core.Environment_core import Environment_core
import numpy as np

class Environment(Environment_core):
    def __init__(self, objective_function, penalty_functions:list = []):
        super().__init__(penalty_functions)
        self.objective_function = objective_function
    
    def get_fitness(self, individual)->float:
        f = self.objective_function(individual.x)
        return f
    
    def evaluate(self, population):
        for individual in population:
            individual.score = individual.fitness + individual.penalty


class FitnessShare(Environment):
    def __init__(self, objective_function, penalty_functions:list = [], alpha:float = 1., d_share:float = 5.):
        super().__init__(objective_function, penalty_functions)
        self.alpha = alpha
        self.d_share = d_share
    
    def evaluate(self, population):
        shd_matrix = np.eye(len(population))

        for i in range(len(population)-1):
            for j in range(i+1, len(population)):
                ind_i = population[i]
                ind_j = population[j]
                d = np.linalg.norm(ind_i.x - ind_j.x)
                shd = 1. - (d/self.d_share)**self.alpha if d <= self.d_share else 0.
                shd_matrix[i,j] = shd
                shd_matrix[j,i] = shd
        
        for idx, individual in enumerate(population):
            individual.score = (individual.fitness + individual.penalty)/np.sum(shd_matrix[idx,:])