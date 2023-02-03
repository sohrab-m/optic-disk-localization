import os
from functools import reduce

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