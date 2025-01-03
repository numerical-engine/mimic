from mimic.core.Environment_core import Environment_core
import numpy as np

class Environment(Environment_core):
    def __init__(self, objective_function, penalty_functions:list = []):
        self.objective_function = objective_function
        self.penalty_functions = penalty_functions
    
    def get_fitness(self, individual)->float:
        f = self.objective_function(individual.x)
        return f
    
    def get_penalty(self, individual)->float:
        p = 0.
        for p_func in self.penalty_functions:
            p += p_func(individual.x)
        return float(p)
    
    def set_score(self, population)->None:
        for individual in population:
            if not individual.already_eval:
                individual.fitness = self.get_fitness(individual)
                individual.penalty = self.get_penalty(individual)
                individual.score = individual.fitness + individual.penalty


class FitnessShare(Environment):
    def __init__(self, objective_function, penalty_functions:list = [], alpha:float = 1., d_share:float = 5.):
        super().__init__(objective_function, penalty_functions)
        self.alpha = alpha
        self.d_share = d_share
    
    def set_score(self, population)->None:
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
            individual.fitness = self.get_fitness(individual)
            individual.penalty = self.get_penalty(individual)
            individual.score = (individual.fitness + individual.penalty)/np.sum(shd_matrix[idx,:])