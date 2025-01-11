import numpy as np
from copy import deepcopy
import networkx as nx
import sys

def squeeze(population, indice:tuple):
    individuals = [population.individuals[idx].copy() for idx in indice]
    return type(population)(individuals, population.generation)

def split(population, num:int, shuffle:bool = True):
    assert (len(population)%num) == 0
    if shuffle:
        population.shuffle()
    indice = np.arange(len(population)).reshape((num, -1))

    populations = [squeeze(population, idx) for idx in indice]
    return populations

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

def elite_set(population, num:int):
    pop_sort = sort(population)
    elite = squeeze(pop_sort, np.arange(num))
    return elite

def concatenate(population1, population2):
    individuals = deepcopy(population1.individuals) + deepcopy(population2.individuals)
    generation = max(population1.generation, population2.generation)

    return type(population1)(individuals, generation)

def bconcatenate(populations:list):
    individuals = []
    for population in populations:
        individuals += deepcopy(population.individuals)
    generation = max([population.generation for population in populations])

    return type(populations[0])(individuals, generation)


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
    score = np.array([individual.score for individual in population])

    for curr_idx, individual in zip(np.arange(num_node, dtype=int), population):
        x = individual.x
        s = individual.score
        
        better_idx = np.where(s < score)[0]
        if len(better_idx) == 0: continue
        better_pops = np.stack([population[idx].x for idx in better_idx], axis = 0)
        nearest_idx = better_idx[np.argmin(np.linalg.norm(better_pops - x, axis = 1))]

        Adj_matrix[curr_idx, nearest_idx] = np.linalg.norm(population[nearest_idx].x - x)
        Adj_matrix[nearest_idx, curr_idx] = np.linalg.norm(population[nearest_idx].x - x)
    
    mean_edge_length = np.mean(Adj_matrix[Adj_matrix > 0.])
    Adj_matrix[Adj_matrix > (phi*mean_edge_length)] = 0.
    graph = nx.from_numpy_array(Adj_matrix)
    graphs = nx.components.connected_components(graph)

    cluster = [tuple(g) for g in graphs]
    
    return cluster


def NBC2(population, phi:float = 2., b:float = None)->list[tuple[int]]:
    num_node = len(population)
    dim = len(population[0].x)
    score = np.array([individual.score for individual in population])
    if b is None:
        b = (-4.69e-4*dim**2 + 0.0263*dim + 3.66/dim - 0.457)*np.log10(num_node) + 7.51e-4*dim**2 - 0.0421*dim - 2.26/dim + 1.83
    Adj_matrix = np.zeros((num_node, num_node)) #Adjacency matrix whcih non-zero values mean distance between node

    for curr_idx, individual in zip(np.arange(num_node, dtype=int), population):
        x = individual.x
        s = individual.score

        better_idx = np.where(s < score)[0]
        if len(better_idx) == 0: continue
        better_pops = np.stack([population[idx].x for idx in better_idx], axis = 0)

        nearest_idx = better_idx[np.argmin(np.linalg.norm(better_pops - x, axis = 1))]

        Adj_matrix[curr_idx, nearest_idx] = np.linalg.norm(population[nearest_idx].x - x)
    
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

def is_nonDominated(population, i:int, comp:np.ndarray = None)->bool:
    if comp is None: comp = np.arange(len(population))
    if len(comp) == 0: return True
    
    target_fitness = population[i].fitness
    fitness = np.stack([population[c].fitness for c in comp], axis = 0)
    num, M = fitness.shape
    mask = np.zeros(num, dtype=int)

    for m in range(M):
        mask += (target_fitness[m] <= fitness[:,m]).astype(int)
    
    if np.any(mask == 0):
        return False
    else:
        return True

def get_frontRank(population)->np.ndarray:
    comp = np.arange(len(population))
    frontRank = np.array([np.nan]*len(population))
    current_rank = 0

    while len(comp) > 0:
        non_dominated = np.array([is_nonDominated(population, c, comp) for c in comp])

        non_dominated_idx = np.array([c for c, nd in zip(comp, non_dominated) if nd])
        frontRank[non_dominated_idx] = current_rank
        current_rank += 1

        comp = comp[non_dominated == False]
    
    return frontRank.astype(int)