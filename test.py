import mimic
import numpy as np
import sys

class Optim(mimic.core.Optimizer_meta):
    def __init__(self):
        self.selection = mimic.selection.tournament()
        self.crossover = mimic.crossover.blx_alpha(3)
        self.mutation = mimic.mutation.normal(0.001)
        self.survival = mimic.survival.mu_to_lam()
    def run(self, population, environment):
        parents = self.selection(population, 100)
        offspring = self.crossover(parents)
        offspring = self.mutation(offspring)
        population = self.survival(parents, offspring, environment)

        return population

obj_func = mimic.benchmark.Rastrigin(2)
x_init = np.random.uniform(-5.12, 5.12, (100,2))
penalty_u = mimic.Penalty_Upper(5.12*np.ones(2))
penalty_l = mimic.Penalty_Lower(-5.12*np.ones(2))
individuals = [mimic.Individual(xi) for xi in x_init]
environment = mimic.Environment(obj_func, penalty_functions = [penalty_l, penalty_u])
population = mimic.Population(individuals)
optim = Optim()

environment.set_score(population)
generation = 500
for _ in range(generation):
    population = optim(population, environment)
    print(mimic.utils.population.get_elite(population))