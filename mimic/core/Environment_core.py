class Environment_core:
    def get_fitness(self, individual):
        raise NotImplementedError
    def get_penalty(self, individual):
        raise NotImplementedError
    def set_score(self, population):
        raise NotImplementedError