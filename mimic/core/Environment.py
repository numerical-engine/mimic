from typing import Union

class Environment:
    """Abstruct class for environment

    Args:
        objective_function (Function.Function_meta): Objective function class.
        penalty_functions (list[Function.Penalty_function]): list of penalty function class.
        att_functions (list[Function.Function_meta]) Other functions for score.
    Attributes:
        objective_function (Function.Function_meta): Objective function class.
        penalty_functions (list[Function.Penalty_function]): list of penalty function class.
        att_functions (list[Function.Function_meta]) Other functions for score.
    """
    def __init__(self, objective_function, penalty_functions:list = [], att_functions:list = []):
        self.objective_function = objective_function
        self.penalty_functions = penalty_functions
        self.att_functions = att_functions
    
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
    
    def get_score(self, individual, sum:bool = True)->Union[float, tuple]:
        """Return score

        Args:
            individual (core.Individual.Individual): Individual
            sum (bool, optional): If true, return the sum of fitness, penalty, and att functions.If False, return each values. Defaults to True.

        Returns:
            Union[float, tuple]: score values
        """
        f = self.get_fitness(individual)
        p = self.get_penalty(individual)
        s = 0.
        for att_func in self.att_functions:
            s += att_func(individual)
        
        if sum:
            return float(s + p + f)
        else:
            return float(f), float(p), float(s)