import numpy as np

class Crossover_meta:
    """Class for crossover and generating population of offspring

    Args:
        parent_num (int): Number of parents for crossover.
        offspring_num (int): Number of offsprings given from one crossover.
    Attributes:
        parent_num (int): Number of parents for crossover.
        offspring_num (int): Number of offsprings given from one crossover.
    """
    def __init__(self, offspring_num:int = 2, parent_num:int = 2):
        self.parent_num = parent_num
        self.offspring_num = offspring_num
    def make_pairs(self, parents)->list[list]:
        """Returns individual pairs for recombination

        Args:
            parents (core.Population.Population): Population of parents
        Returns:
            list[list[core.Individual.Individual]]: list of pairs. length "L" equals to number of pairs.
            Therefore, this crossover generate "offspring_num*L" offsprings.
            Each element of output is a list of pair which length equals to "parent_num".
        """
        assert (len(parents) % self.parent_num) == 0
        indice = np.arange(len(parents)).astype(int)
        np.random.shuffle(indice)

        indice = indice.reshape((-1, self.parent_num))
        return [[parents[i].copy() for i in idx] for idx in indice]
    
    def __call__(self, parents):
        """Returns population of offspring

        Args:
            parents (core.Population.Population): Population of parents
            environment (core.Environment.Environment): Environment
        Returns:
            core.Population.Population: Population of offsprings
        """
        pair_list = self.make_pairs(parents) #list[list[Individual]]
        assert len(pair_list[0]) == self.parent_num
        offsprings = [] #list[Individual]
        for pair in pair_list:
            offsprings += self._run(pair)
        
        offsprings = type(parents)(offsprings, parents.generation)

        return offsprings
    
    def _run(self, pair:list)->list:
        """Returns offsprings given from pair.

        Args:
            pair (list[core.Individual.Individual]): pair of individual
            environment (core.Environment.Environment): Environment
        Returns:
            list[core.Individual.Individual]: list of individual about offspring
        """
        x_pair = tuple(individual.x for individual in pair)
        x_offs = self.run(x_pair)
        assert len(x_offs) == self.offspring_num

        att_offs = [{} for _ in range(self.offspring_num)]
        for key in pair[0].att_keys:
            att_pair = tuple(individual.__dict__[key] for individual in pair)
            for idx, att in enumerate(self.run(att_pair)):
                att_offs[idx][key] = att
        
        offspring = [type(pair[0])(x, att = att) for x, att in zip(x_offs, att_offs)]
        return offspring
    
    def run(self, x_pair:tuple[np.ndarray])->list[np.ndarray]:
        """Generate variables for offspring

        Args:
            x_pair (tuple[np.ndarray]): tuple of variables with respect to pair
        Raises:
            NotImplementedError: _description_
        Returns:
            list[np.ndarray]: list of variables with respect to offspring.
        """
        raise NotImplementedError