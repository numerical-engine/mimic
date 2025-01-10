class Environment_core:
    def __init__(self, penalty_functions:list = []):
        self.penalty_functions = penalty_functions
    
    def get_fitness(self, individual):
        raise NotImplementedError
    
    def get_penalty(self, individual)->float:
        p = 0.
        for p_func in self.penalty_functions:
            p += p_func(individual.x)
        return float(p)
    
    def evaluate(self, population):
        raise NotImplementedError
    
    def set_score(self, population)->None:
        for individual in population:
            if not individual.already_fit:
                individual.fitness = self.get_fitness(individual)
                individual.penalty = self.get_penalty(individual)
        self.evaluate(population)