class optimizer_meta:
    """The class for optimization and population update

    Note:
        This class should be inherited, because method **run** hasn't been defined. Codes in `mimic/method/` will help you.
    """
    def __call__(self, pop, env):
        pop_new = pop.copy()
        #####update generation and age
        pop_new.generation += 1
        for i in range(len(pop_new)):
            pop_new.individuals[i].age += 1
        
        return self.run(pop_new, env)
    
    def run(self, pop, env):
        """Update population

        Args:
            pop (core.Population.population): Current population.
            env (core.Environment.environment): Environment we concern.
        Returns:
            (core.Population.population): New population.
        Raises:
            NotImplementedError: This class should be inherited.
        """
        raise NotImplementedError