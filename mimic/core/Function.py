import numpy as np

class Function_meta:
    """Abstruct class for any function.

    Note:
        Any function should be defined by classes inheriting Function_meta, because forward method isn't implemented.
        Codes in `./mimic/benchmark/` will help you.
    """
    def __call__(self, individual)->float:
        return self.forward(individual)
    def forward(self, individual)->float:
        """Output function value

        Args:
            individual (core.Individual.Individual): Individual.
        Returns:
            float: Function value.
        """
        raise NotImplementedError


class Penalty_function_meta(Function_meta):
    """Inequality penalty function

    Args:
        weight (float): Weight parameter. Defaults to 1.0.
    Attributes:
        weight (float): Weight parameter. Defaults to 1.0.
    Note:
        MIMIC can adapt only **lower than** constraint.
    Note:
        Any penalty function should be defined by classes inheriting Penalty_function_meta, because evaluate method isn't implemented.
    """
    def __init__(self, weight:float = 1.):
        assert weight > 0., f"weight should be greater than zero."
        self.weight = weight
    
    def evaluate(self, individual)->float:
        """Evaluate signed penalty value.

        If a constraint can be expressed as g(x) <= 0, evaluate method returns g(x).
        Args:
            individual (core.Individual.Individual): Individual
        Raises:
            NotImplementedError: This class is abstruct class
        Returns:
            float: Signed penalty value
        """
        raise NotImplementedError

    def forward(self, individual)->float:
        """Output penalty function value

        Args:
            individual (core.Individual.Individual): Individual.
        Returns:
            float: Penalty function value.
        """
        penalty_values = self.weight*self.evaluate(individual) #(float)
        #####If penalty_values < 0, set to 0. because x is feasible.
        return np.max(penalty_values, 0.)


class Penalty_search_domain_upper(Penalty_function_meta):
    """Penalty function for search domain of upper limit

    Args:
        weight (float): Weight parameter. Defaults to 1.0.
        xu (np.ndarray): Upper bound for solution variables.
        att (dict[str, np.ndarray]): Upper bound for additional variables
    Attributes:
        weight (float): Weight parameter. Defaults to 1.0.
        xu (np.ndarray): Upper bound for solution variables.
        att_keys (list[str]): List of names given from att.
        key (np.ndarray): Suplemental upper bound given from att. The name "key" is same with att_keys[i].
    """
    def __init__(self, xu:np.ndarray, weight:float = 1., att:dict = {}):
        super().__init__(weight)
        self.xu = xu
        self.att = att

        self.att_keys = []
        for key in att.keys():
            assert isinstance(att[key], np.ndarray), f"{key} should be numpy"
            assert len(att[key].shape) == 1, f"{key} should be 1st order tensor, but shape is {att[key].shape}"
            self.att_keys.append(key)
            self.__dict__[key] = att[key]
    
    def evaluate(self, individual)->float:
        """Evaluate signed penalty value.

        If a constraint can be expressed as g(x) <= 0, evaluate method returns g(x).
        Args:
            individual (core.Individual.Individual): Individual
        Returns:
            float: Penalty value
        """
        #####calculate penalty values for each solution variables
        p_value = np.sum(np.maximum(individual.x - self.xu, 0.))

        #####calculate penalty values for each additional variables
        for key in self.att_keys:
            p_value += np.sum(np.maximum(individual.__dict__[key] - self.__dict__[key], 0.))
        
        return p_value


class Penalty_search_domain_lower(Penalty_function_meta):
    """Penalty function for search domain of lower limit

    Args:
        weight (float): Weight parameter. Defaults to 1.0.
        xl (np.ndarray): Lower bound for solution variables.
        att (dict[str, np.ndarray]): Lower bound for additional variables
    Attributes:
        weight (float): Weight parameter. Defaults to 1.0.
        xl (np.ndarray): Lower bound for solution variables.
        att_keys (list[str]): List of names given from att.
        key (np.ndarray): Suplemental lower bound given from att. The name "key" is same with att_keys[i].
    """
    def __init__(self, xl:np.ndarray, weight:float = 1., att:dict = {}):
        super().__init__(weight)
        self.xl = xl
        self.att = att

        self.att_keys = []
        for key in att.keys():
            assert isinstance(att[key], np.ndarray), f"{key} should be numpy"
            assert len(att[key].shape) == 1, f"{key} should be 1st order tensor, but shape is {att[key].shape}"
            self.att_keys.append(key)
            self.__dict__[key] = att[key]
    
    def evaluate(self, individual)->float:
        """Evaluate signed penalty value.

        If a constraint can be expressed as g(x) <= 0, evaluate method returns g(x).
        Args:
            individual (core.Individual.Individual): Individual
        Returns:
            float: Penalty value
        """
        #####calculate penalty values for each solution variables
        p_value = np.sum(np.maximum(individual.x - self.xl, 0.))

        #####calculate penalty values for each additional variables
        for key in self.att_keys:
            p_value += np.sum(np.maximum(individual.__dict__[key] - self.__dict__[key], 0.))
        
        return p_value