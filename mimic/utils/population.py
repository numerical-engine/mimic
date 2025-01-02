import numpy as np
from copy import deepcopy
import networkx as nx

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

def NBC1(population, phi:float = 2.)->list[tuple[int]]:
    num_node = len(population)
    Adj_matrix = np.zeros((num_node, num_node))
    fitness = np.array([individual.fitness for individual in population])

    for curr_idx, individual in zip(np.arange(num_node, dtype=int), population):
        x = individual.x
        f = individual.fitness
        
        better_idx = np.where(f < fitness)[0]
        if len(better_idx) == 0: continue

        better_pops = population[better_idx]
        nearest_idx = better_idx[np.argmin(np.linalg.norm(better_pops - x, axis = 1))]

        Adj_matrix[curr_idx, nearest_idx] = np.linalg.norm(population[nearest_idx] - x)
        Adj_matrix[nearest_idx, curr_idx] = np.linalg.norm(population[nearest_idx] - x)
    
    mean_edge_length = np.mean(Adj_matrix[Adj_matrix > 0.])
    Adj_matrix[Adj_matrix > (phi*mean_edge_length)] = 0.
    graph = nx.from_numpy_array(Adj_matrix)
    graphs = nx.components.connected_components(graph)

    cluster = [tuple(g) for g in graphs]
    
    return cluster


def NBC2(population, phi:float = 2., b:float = None)->list[tuple[int]]:
    num_node, dim = population.shape
    fitness = np.array([individual.fitness for individual in population])
    if b is None:
        b = (-4.69e-4*dim**2 + 0.0263*dim + 3.66/dim - 0.457)*np.log10(num_node) + 7.51e-4*dim**2 - 0.0421*dim - 2.26/dim + 1.83
    Adj_matrix = np.zeros((num_node, num_node)) #Adjacency matrix whcih non-zero values mean distance between node

    for curr_idx, individual in zip(np.arange(num_node, dtype=int), population):
        x = individual.x
        f = individual.fitness
        
        better_idx = np.where(f < fitness)[0]
        if len(better_idx) == 0: continue

        better_pops = population[better_idx]
        nearest_idx = better_idx[np.argmin(np.linalg.norm(better_pops - x, axis = 1))]

        Adj_matrix[curr_idx, nearest_idx] = np.linalg.norm(population[nearest_idx] - x)
    
    mean_edge_length = np.mean(Adj_matrix[Adj_matrix > 0.])
    Adj_matrix[Adj_matrix > (phi*mean_edge_length)] = 0.

    for idx in range(num_node):
        out_edge = Adj_matrix[idx]; out_edge = out_edge[out_edge > 0.]
        in_edge = Adj_matrix[:,idx]; in_edge = in_edge[in_edge > 0.]

        if (len(out_edge) == 1) & (len(in_edge) > 2):
            out_len = out_edge[0]
            in_len = np.mean(in_edge)

            if (out_len/in_len) > b:
                Adj_matrix[idx] = 0.
    
    graph = (nx.from_numpy_array(Adj_matrix)).to_undirected()
    graphs = nx.components.connected_components(graph)

    cluster = [tuple(g) for g in graphs]
    
    return cluster