class Optimizer:
    def __call__(self, population, environment):
        if not population.already_eval:
            environment.set_score(population)
            
        population_new = population.copy()
        population_new.generation += 1
        for i in range(len(population_new)):
            population_new.individuals[i].age += 1
        
        population_new = self.run(population_new, environment)
        if not population_new.already_eval:
            environment.set_score(population_new)
        return population_new
    
    def run(self, population, environment):
        raise NotImplementedError