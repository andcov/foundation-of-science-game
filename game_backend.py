import numpy as np
import random

# from https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def nparr_to_list(arr):
    return [int(i) for i in arr]
# ==============================================================

class Level():
    known_points = {}

    def __init__(self):
        raise NotImplemented
    
    def description(self): # prints a description of the level,
        # in particular how many dimensions and how the context of the model looks like
        raise NotImplemented

    def move(self, movement_vector):
        raise NotImplemented

    def save_point(self, name):
        raise NotImplemented
    
    def measure_angle(self, left_point, right_point): # measuring the angle between two points and the current position
        raise NotImplemented
    
    def check(self, model): # odel is a function that given the context (i.e. the position and where to move) and predicts how a state (i.e. the position) changes
        raise NotImplemented

class Euclidean(Level):
    def __init__(self, dim: int = 3):
        self.dim = dim
        self.position = np.zeros(dim)
    
    def description(self):
        return """This level takes dim (usually 3) values as a movementvector and
        expects the model to take a dim sized list position and a dim sized list movement_vector
        it should return a dim sized list with the predicted new position
        
        so model should have type model(position: List(int), movement: List(int)) -> List(int) where every list is dim long"""
    
    def move(self, movement_vector: np.array):
        self.position += movement_vector
    
    def save_point(self, name: str):
        self.known_points[name] = self.position.copy()

    def measure_angle(self, left_point: str, right_point: str) -> int: # measuring the angle between two points and the current position
        a = self.known_points[left_point] - self.position
        b = self.known_points[right_point] - self.position
        return angle_between(a, b)

    def measure_length(self, other_point) -> int:
        return self.known_points[other_point]-self.position

    def check(self, model):
        for i in range(100):
            pos = np.random.randint(-1000, 1000, self.dim)
            move = np.random.randint(-1000, 1000, self.dim)
            if nparr_to_list(pos+move) != model(nparr_to_list(pos), nparr_to_list(move)):
                return False
        return True

