class Optimizer_meta:
    """The class for optimization and population update

    Note:
        This class should be inherited, because method **run** hasn't been defined. Codes in `mimic/method/` will help you.
    """
    def __call__(self, population, environment):
        population_new = population.copy()
        #####update generation and age
        population_new.generation += 1
        for i in range(len(population_new)):
            population_new.individuals[i].age += 1
        
        population_new = self.run(population_new, environment)
        if not population_new.already_eval:
            environment.set_score(population_new)
        return population_new
    
    def run(self, population, environment):
        """Update population

        Args:
            population core.Population.Population: Current population.
            environment core.Environment.Environment: Environment we concern.
        Returns:
            core.Population.Population: New population.
        Raises:
            NotImplementedError: This class should be inherited.
        """
        raise NotImplementedError