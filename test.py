import mimic
from mimic import SOEA
import numpy as np
import sys


class Optim(mimic.Optimizer):
    def __init__(self):
        self.selection = mimic.selection.tournament()
    def run(self, population, environment):
        parents = self.selection(population, 100)
        print(len(parents))
        sys.exit()

x_init = np.random.uniform(1., 5., (100,2))
individuals = [SOEA.Individual(xi) for xi in x_init]
population = mimic.Population(individuals)
obj_func = mimic.benchmark.sphere()
environment = SOEA.Environment(obj_func)
environment.set_score(population)
optim = Optim()

generation = 500
for _ in range(generation):
    population = optim(population, environment)
    sys.exit()