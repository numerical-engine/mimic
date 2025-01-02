import numpy as np
from mimic import CMAES
import mimic
import mimic.benchmark
import sys

mean = 5.*np.ones(2)
sigma = 1.
population = CMAES.Population(mean, sigma)
print(population.c_mu)
# optim = CMAES.Optimizer()
# func = mimic.benchmark.sphere()
# environment = CMAES.Environment(func)

# for i in range(100):
#     population = optim(population, environment)
#     print(population.mean)