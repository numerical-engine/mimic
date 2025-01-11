import numpy as np
import mimic
from mimic.MOEA.Model import MOGA
from mimic import MOEA
import sys
import pandas as pd
import grafi

import mimic.utils


class Optim(mimic.Optimizer):
    def __init__(self):
        self.selection = mimic.selection.fps()
        self.crossover = mimic.crossover.blx_alpha()
        # self.mutation = mimic.mutation.pbm(np.array([0.1, 0.]), np.array([1., 5.]), eta = 40.)
        self.mutation = mimic.mutation.normal(0.01)
    def run(self, populations, environment):
        parents = self.selection(population, 100)
        offspring = self.crossover(parents)
        offspring = self.mutation(offspring)

        return offspring


x_lower = mimic.PenaltyLower(np.array([0.1, 0.]), 100.)
x_upper = mimic.PenaltyUpper(np.array([1., 5.]), 100.)

class obj_func1(mimic.Function):
    def forward(self, x:np.ndarray)->float:
        return x[0]
class obj_func2(mimic.Function):
    def forward(self, x:np.ndarray)->float:
        return (x[1]+1.)/x[0]

obj_funcs = [obj_func1(), obj_func2()]

d_share = 1.
fl = np.array([0.1, 1.])
fu = np.array([1., 10.])
environment = MOGA.Environment_MOGA(obj_funcs, d_share, fl, fu, [x_lower, x_upper])
x0 = np.random.uniform(0.1, 1., 100)
x1 = np.random.uniform(0., 5., 100)
X = np.stack((x0, x1), axis = 1)

individuals = [MOEA.Individual(x) for x in X]
population = mimic.Population(individuals)

optim = Optim()

for _ in range(100):
    population = optim(population, environment)

# individuals = population.individuals
# F = []
# X = []
# for individual in population.individuals:
#     f1 = obj_funcs[0](individual.x)
#     f2 = obj_funcs[1](individual.x)
#     X.append(individual.x)
#     F.append(np.array([f1, f2]))

# X = np.stack(X, axis = 0)
# F = np.stack(F, axis = 0)
# F = pd.DataFrame(F)
# F.to_csv("F.csv")
# X = pd.DataFrame(X)
# X.to_csv("X.csv")