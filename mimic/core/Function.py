import numpy as np
import sys

class Function_meta:
    """Abstruct class for any function.

    Note:
        Any function should be defined by classes inheriting Function_meta, because "forward" method isn't implemented.
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
    
    def __call__(self, individual)->float:
        #####If penalty_values < 0, set to 0. because x is feasible.
        penalty_values = np.max([self.forward(individual), 0.])
        return self.weight*penalty_values
    
    def forward(self, individual)->float:
        """Evaluate signed penalty value.

        If a constraint can be expressed as g(x) <= 0, forward method returns g(x).
        Args:
            individual (core.Individual.Individual): Individual
        Raises:
            NotImplementedError: This class is abstruct class
        Returns:
            float: Signed penalty value
        """
        raise NotImplementedError


class Penalty_Lower(Penalty_function_meta):
    """Penalty function for search domain of lower limit

    Args:
        weight (float): Weight parameter. Defaults to 1.0.
        xl (np.ndarray): Lower bound for solution variables.
    Attributes:
        weight (float): Weight parameter. Defaults to 1.0.
        xl (np.ndarray): Lower bound for solution variables.
    """
    def __init__(self, xl:np.ndarray, weight:float = 1.):
        super().__init__(weight)
        self.xl = xl
    
    def forward(self, individual)->float:
        """Evaluate signed penalty value.

        If a constraint can be expressed as g(x) <= 0, forward method returns g(x).
        Args:
            individual (core.Individual.Individual): Individual
        Returns:
            float: Signed penalty value
        """
        return np.sum(np.maximum(self.xl - individual.x, 0.))

class Penalty_Upper(Penalty_function_meta):
    """Penalty function for search domain of lower limit

    Args:
        weight (float): Weight parameter. Defaults to 1.0.
        xu (np.ndarray): Upper bound for solution variables.
    Attributes:
        weight (float): Weight parameter. Defaults to 1.0.
        xu (np.ndarray): Upper bound for solution variables.
    """
    def __init__(self, xu:np.ndarray, weight:float = 1.):
        super().__init__(weight)
        self.xu = xu
        
    def forward(self, individual)->float:
        """Evaluate signed penalty value.

        If a constraint can be expressed as g(x) <= 0, forward method returns g(x).
        Args:
            individual (core.Individual.Individual): Individual
        Returns:
            float: Signed penalty value
        """
        return np.sum(np.maximum(individual.x - self.xu, 0.))