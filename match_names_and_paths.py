import os
import csv
import scipy.io as sio

def find_file(filename, directory):
    for root, _, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    return None


# List of filenames to search for
filenames = sio.loadmat("/home/synrg/Documents/Sohrab/optic-disk-localization/dataset/annotations/11_44/annotations.mat")['names'][0]
filenames = [filename[0] for filename in filenames]
values = sio.loadmat("/home/synrg/Documents/Sohrab/optic-disk-localization/dataset/annotations/11_44/annotations.mat")['values'][0:2, :]

directory = "/home/synrg/Documents/Sohrab/optic-disk-localization/dataset"

# Save results to a CSV file
with open("annotations.csv", "w") as f:
    writer = csv.writer(f)
    for idx, filename in enumerate(filenames): 
        path = find_file(filename, directory)
        if path is not None:
            writer.writerow((filename, values[0, idx], values[1, idx], path))

