class environment:
    """Abstruct class for environment

    Args:
        objective_function (Function.function_meta): Objective function class.
        penalty_functions (list[Function.penalty_function]): list of penalty function class.
        att_functions (list[Function.function_meta]) Other functions for score.
    Attributes:
        objective_function (Function.function_meta): Objective function class.
        penalty_functions (list[Function.penalty_function]): list of penalty function class.
        att_functions (list[Function.function_meta]) Other functions for score.
    """
    def __init__(self, objective_function, penalty_functions:list = [], att_functions:list = []):
        self.objective_function = objective_function
        self.penalty_functions = penalty_functions
        self.att_functions = att_functions
    
    def get_fitness(self, individual)->float:
        """Return fitness value of individual

        Args:
            individual (core.Individual.individual): Individual
        Returns:
            float: Fitness value
        """
        f = self.objective_function(individual.x)
        return f
    
    def get_penalty(self, individual)->float:
        """Return penalty value of individual

        Args:
            individual (core.Individual.individual): Individual
        Returns:
            float: penalty value
        """
        p = 0.
        for p_func in self.penalty_functions:
            p += p_func(individual.x)
        return float(p)
    
    def get_score(self, individual, sum = True)->float:
        """Return score of individual

        Args:
            individual (core.Individual.individual): Individual
        Returns:
            float: score
        """
        f = self.get_fitness(individual)
        p = self.get_penalty(individual)
        s = 0.
        for att_func in self.att_functions:
            s += att_func(individual.x)
        
        if sum:
            return float(s + p + f)
        else:
            return float(f), float(p), float(s)