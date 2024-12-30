from typing import Union
import numpy as np

class Environment:
    """Abstruct class for environment

    Args:
        objective_function (Function.Function_meta): Objective function class.
        penalty_functions (list[Function.Penalty_function]): list of penalty function class.
    Attributes:
        objective_function (Function.Function_meta): Objective function class.
        penalty_functions (list[Function.Penalty_function]): list of penalty function class.
    """
    def __init__(self, objective_function, penalty_functions:list = []):
        self.objective_function = objective_function
        self.penalty_functions = penalty_functions
    
    def get_fitness(self, individual)->float:
        """Return fitness value of individual

        Args:
            individual (core.Individual.Individual): Individual
        Returns:
            float: Fitness value
        """
        f = self.objective_function(individual)
        return f
    
    def get_penalty(self, individual)->float:
        """Return penalty value of individual

        Args:
            individual (core.Individual.Individual): Individual
        Returns:
            float: penalty value
        """
        p = 0.
        for p_func in self.penalty_functions:
            p += p_func(individual)
        return float(p)
    
    def set_score(self, population):
        """Return score

        Args:
            population (core.Population.Population): Population
        """
        for individual in population:
            if not individual.already_eval:
                individual.fitness = self.get_fitness(individual)
                individual.penalty = self.get_penalty(individual)
                individual.score = individual.fitness + individual.penalty