import os
from functools import reduce
import numpy as np

def check_path_exists(path):
    assert os.path.exists(path), f"Path '{path}' does not exist."
    return os.path.exists(path)


def check_paths_exist(paths):
    for path in paths:
        check_path_exists(path)


def flatten_list(lists):
    return reduce(list.__add__, lists)

def find_indices(a_list, item_to_find):
    indices = []
    for idx, value in enumerate(a_list):
        if value == item_to_find:
            indices.append(idx)
    return indices

if __name__ == "__main__":
    check_path_exists('/mnt/c/Users/ssohr/OneDrive/Documents/opti-disk-localization')




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
    return np.dot(v1_u, v2_u)