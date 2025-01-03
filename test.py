import numpy as np
import mimic
from mimic import SOEA
from matplotlib.figure import Figure
from matplotlib import cm
from mimic.core.Function_core import Function
import mimic.utils
import sys

def him(x):
    xx, yy = x[0], x[1]
    return (xx**2 + yy - 11.)**2 + (xx + yy**2 - 7.)**2

x = np.linspace(-5., 5., 1000)
y = np.linspace(-5., 5., 1000)
xx, yy = np.meshgrid(x, y)
value = him([xx, yy])
figure = Figure(figsize = (int(7), int(7)), dpi = 100)
plot = figure.add_subplot(111)
level = [0.5, 5., 10., 20., 40., 60.]
plot.contour(xx, yy, value, levels = level, cmap = "gray")

pop_num = 200

class Optim(mimic.Optimizer):
    def __init__(self):
        # self.selection = mimic.selection.ranking(mimic.utils.Selection.roulette_wheel)
        self.selection = mimic.selection.fps()
        self.crossover = mimic.crossover.sbx(np.array([-4., -4.]), np.array([4., 4.]), eta = 1.)
        # self.mutation = mimic.mutation.pbm(np.array([-4., -4.]), np.array([4., 4.]), eta = 50.)
        self.mutation = mimic.mutation.normal(0.1)
        self.survival = mimic.survival.mu_to_lam()
    
    def run(self, population, environment):
        pop_num = len(population)
        parents = self.selection(population, pop_num*2)
        offspring = self.crossover(parents)
        offspring = self.mutation(offspring)
        population = self.survival(parents, offspring, environment, fource = True)
        population = mimic.utils.Population.elite_set(population, pop_num)

        return population


np.random.seed(5)
x = np.random.uniform(-4., 4., (pop_num, 2))
color = ("magenta", "cyan", "lime", "yellow", "grey", "red", "orange")

individuals = [SOEA.Individual(_x) for _x in x]
population = mimic.Population(individuals)
environment = SOEA.FitnessShare(mimic.benchmark.himmelblau(), alpha = 1., d_share = 0.1)
# environment = SOEA.Environment(mimic.benchmark.himmelblau())
environment.set_score(population)
optim = Optim()
for i in range(50):
    # print(i)
    population = optim(population, environment)

X = np.stack([individual.x for individual in population], axis = 0)
plot.scatter(X[:,0], X[:,1], s = 100., c = "cyan", edgecolors="black", marker="o", linewidths=2)

figure.savefig("aaa.jpg")