class SampleClass:
    """Just a sample class
    This discription are called google style.

    Parameters
        :x (int): input x
        :y (float): input y
    
    Attribution
        :x (int): attribution
        :y (float): attribution2.
    
    Method
        :sum: one line summary
    """
    def __init__(self, x:int, y:float = 0.):
        self.x = x
        self.y = y
    def sum(self, a:float)->float:
        """sum
        Args:
            a (float): sample args
        Returns:
            float: sum
        """
        return self.x + self.y + a