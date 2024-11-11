import numpy as np
from copy import deepcopy

def squeeze(population, indice:tuple):
    """Squeeze individuals from population.

    Args:
        population (core.Population.Population): Population
        indice (tuple): tuple of index of which individuals are squeezed
    Returns:
        core.Population.Population: Squeezed population
    """
    individuals = [population.individuals[idx].copy() for idx in indice]
    return type(population)(individuals, population.generation)


def get_elite(population, eval_fitness:bool = True, only_feasible:bool = True):
    """Return a elite individual (=the best fitness individual)

    Args:
        population (core.Population.Population): Population
        eval_fitness (bool, optional): If True, elite is selected by fitness. Otherwise, selected by score. Defaults to True.
        only_feasible (bool, optional): If True, only feasible individuals are evaluated. Defaults to True.
    Returns:
        core.Individual.Individual: Elite.
    Note:
        Return only one of individuals which fitness(or score) is the lowest of all.
    """
    if eval_fitness:
        values = np.array([individual.fitness for individual in population])
    else:
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
    """Concatenate two populations

    Args:
        population1 (core.Population.Population): Population
        population2 (core.Population.Population): Population
    Returns:
        core.Population.Population: Concatenated population
    """
    individuals = deepcopy(population1.individuals) + deepcopy(population2.individuals)
    generation = max(population1.generation, population2.generation)

    return type(population1)(individuals, generation = generation)

def age_sort(population):
    """Sort individuals in population with respect to age

    Args:
        population (core.Population.Population): Population
    Returns:
        Population: Sorted population
    """

    age = np.array([individual.age for individual in population])
    indice = np.argsort(age)

    return squeeze(population, indice)

def sort(population, eval_fitness:bool = True):
    """Sort individuals in population.

    Args:
        population (core.Population.Population): Population.
        eval_fitness (bool, optional): If True, elite is selected by fitness. Otherwise, selected by score. Defaults to True.
    Returns:
        core.Population.Population: Sorted population
    """
    score = np.array([individual.score for individual in population])
    indice = np.argsort(score)
    return squeeze(population, indice)