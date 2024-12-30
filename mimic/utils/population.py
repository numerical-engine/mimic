import numpy as np
from copy import deepcopy

def squeeze(population, indice:tuple):
    individuals = [population.individuals[idx].copy() for idx in indice]
    return type(population)(individuals, population.generation)


def get_elite(population, only_feasible:bool = True):
    assert population.already_eval
    values = np.array([individual.score for individual in population])

    if only_feasible:
        feasibility = np.array([individual.feasible for individual in population])
        for idx in np.argsort(values):
            if feasibility[idx]:
                return population[idx]
    else:
        idx = np.argmin(values)
        return population[idx]

def concatenate(population1, population2):
    individuals = deepcopy(population1.individuals) + deepcopy(population2.individuals)
    generation = max(population1.generation, population2.generation)

    return type(population1)(individuals, generation)

def age_sort(population):
    age = np.array([individual.age for individual in population])
    indice = np.argsort(age)

    return squeeze(population, indice)

def sort(population):
    values = np.array([individual.score for individual in population])
    indice = np.argsort(values)
    return squeeze(population, indice)