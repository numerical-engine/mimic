import numpy as np
from mimic.MOEA.Environment import Environment
from mimic import Optimizer
import mimic
import sys

class Environment_VEGA(Environment):
    def __init__(self, objective_functions:list, idx:int, penalty_functions:list = []):
        assert 0 <= idx < len(objective_functions)
        super().__init__(objective_functions, penalty_functions)
        self.idx = idx
    
    def evaluate(self, population):
        for individual in population:
            individual.score = individual.fitness[self.idx] + individual.penalty


def get_environment_vega(objective_functions:list, penalty_functions:list = []):
    return [Environment_VEGA(objective_functions, idx, penalty_functions) for idx in range(len(objective_functions))]

class Optimizer_VEGA(Optimizer):
    def __init__(self, selection):
        self.selection = selection
    def __call__(self, population, environments:list, parent_num:int):
        obj_num = len(environments)
        populations = mimic.utils.Population.split(population, obj_num, True)
        for environment, population in zip(environments, populations):
            environment.set_score(population)
        
        populations_new = []
        for population in populations:
            population_new = population.copy()
            population_new.generation += 1
            for i in range(len(population_new)):
                population_new.individuals[i].age += 1
            populations_new.append(population_new)

        populations_new = self.run(populations_new, parent_num)
        return populations_new
    
    def run(self, populations:list, parent_num:int):
        parents = []
        for population in populations:
            parent = self.selection(population, parent_num)
            parents.append(parent)
        parents = mimic.utils.Population.bconcatenate(parents)
        return parents