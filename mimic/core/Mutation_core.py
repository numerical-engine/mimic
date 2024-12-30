import numpy as np

class Mutation:
    def __call__(self, population):
        population_new = population.copy()
        population_new.individuals = [self.run(individual) for individual in population_new]
        return population_new
    
    def run(self, individual):
        raise NotImplementedError