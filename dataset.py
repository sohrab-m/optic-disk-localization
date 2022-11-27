import os
import glob
import utils
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torch
import torch.nn as nn
import imageio
from torchvision.utils import make_grid
from torchvision.utils import save_image
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
import random
import pandas as dp
from pathlib import Path
from natsort import natsorted
import pickle
from numpy.linalg import norm
import cv2
import torch.utils.data as data_utils

class BaseDataset(Dataset):
    def __init__(self, mode, img_type, path):
        super().__init__()
        self.mode = mode
        self.img_type = "." + img_type
        self.path = path
        self.files = natsorted(glob.glob(os.path.join(self.path, "*" + self.img_type)))

    def __get_item__(self, idx):
        pass

    @staticmethod
    def img_diff(img1, img2):
        return abs(img1-img2)

class SimpleDataset(BaseDataset):
    def __init__(self, mode, img_type):
        super().__init__(mode, img_type)

        self.csv = None
        self.labels = None
        self.cols = None
        self.valid_distance = None
        self.valid_pixels = None
        self.img_center = None
        self.patch_mask = None
        self.curr_name = None

        # harcoded values we'll have to change later.
        self.img_diam = 1390
        self.patch_diam = 155
        self.x_range = [20,-14]
        self.y_range = [372,-382]
        self.n_patch = 10
    

        self.initialize()

    def initialize(self):
        self.load_labels()
        self.valid_distances()
        self.create_patch_mask()
    
    def create_patch_mask(self):
        patch_rad = int(self.patch_diam/2)
        self.patch_mask = np.asarray([[1  if (x-patch_rad)**2 + (y-patch_rad)**2 < patch_rad ** 2 else 0 \
         for x in range(patch_rad*2)] for y in range(patch_rad*2)])
        self.patch_mask = np.expand_dims(self.patch_mask, -1)

    def valid_distances(self):
        self.valid_distance = int((self.img_diam - self.patch_diam)/2)
        self.img_center = int(self.img_diam/2)
        self.valid_pixels = [[x, y] for x in range(self.img_diam) for y in range(self.img_diam) if self.is_valid(x, y)]
        valid_mask = [[128 if self.is_valid(x, y) else 10 for x in range(self.img_diam)] for y in range(self.img_diam)]
        imageio.imwrite("valid.jpg", np.asarray(valid_mask, dtype=np.uint8))

    def crop_image(self, image):
        return image[self.x_range[0]:self.x_range[1], self.y_range[0]:self.y_range[1], :]

    def load_image(self, idx):
        path = self.files[idx]
        image = self.crop_image(imageio.imread(path))
        return {'path': path, 'img': image, 'labels': self.labels.values[idx]}

    def load_labels(self):
        self.labels = utils.load_csv(self.csv)
        self.cols = list(self.labels.columns)

    def find_files(self):
        self.files, self.csv = SimpleDataset.load_files(self.mode)
        self.files = natsorted(self.files)

    def __getitem__(self, idx):
        self.curr_name = Path(self.files[idx]).stem
        return self.load_image(idx)

    def get_centers(self):
        return [random.choice(self.valid_pixels) for _ in range(self.n_patch)]
    
    def get_patch(self, image):
        centers = self.get_centers()
        if not os.path.exists(self.curr_name):
            os.mkdir(self.curr_name)
        imageio.imwrite(os.path.join(self.curr_name, self.curr_name + self.img_type), image)
        for cx, cy in centers:
            patch_rad = int(self.patch_diam/2)
            patch = image[cx-patch_rad:cx+patch_rad, cy-patch_rad:cy+patch_rad, :] * self.patch_mask
            patch = np.asarray(patch, dtype=np.uint8)
            imageio.imwrite(os.path.join(self.curr_name, f'{cx:04d}_{cy:04d}_{self.curr_name:s}{self.img_type}'), patch)
            print(f"done with {self.curr_name:s}, with {cx:04d} and {cy:04d}")

    def is_valid(self, x, y):
        return (x-self.img_center)**2 + (y-self.img_center)**2 < self.valid_distance**2

    @staticmethod
    def path_maker(mode):
        mode = mode.lower()
        path1, path2 = None, None
        if 'train' in mode:
            path1 = os.path.join('.', 'dataset', 'Training_Set', 'Training', '*.png')
            path2 = os.path.join('.', 'dataset', 'Training_Set', 'RFMiD_Training_Labels.csv')
        elif 'test' in mode:
            path1 =  os.path.join('.', 'dataset', 'Test_Set', 'Test', '*.png')
            path2 = os.path.join('.', 'dataset', 'Training_Set', 'RFMiD_Testing_Labels.csv')
        elif 'val' in mode: 
            path1 =  os.path.join('.', 'dataset', 'Evaluation_Set', 'Validation', '*.png')
            path2 = os.path.join('.', 'dataset', 'Training_Set', 'RFMiD_Validation_Labels.csv')

        return path1, path2

    @staticmethod
    def load_files(mode):
        path1, path2 = MESSIDORDataset.path_maker(mode)
        return glob.glob(path1), path2

class MESSIDORDataset(BaseDataset):
    def __init__(self, mode, img_type, dataroot):
        super().__init__(mode, img_type, dataroot)

        self.valid_distance = None
        self.valid_pixels = None
        self.img_center = None
        self.patch_mask = None
        self.curr_name = None
        self.gt_path = None

        self.prime_files = []
        self.annot_files = []
        self.annots = []

        # harcoded values we'll have to change later.
        self.img_diam = 1377
        self.patch_diam = 153
        self.x_range = [55,1432]
        self.y_range = [428,1805]
        self.n_patch = 20
        self.img_size = (2240, 1488)
    
        self.initialize()

    def initialize(self):
        self.create_gt_path()
        self.separate_files()
        self.process_annot()
        self.valid_distances()
        self.create_patch_mask()

    def create_gt_path(self):
        self.gt_path = os.path.join(self.path, "..", "GT")
        assert os.path.exists(self.gt_path), "Ground truth path does not exist."
    
    def separate_files(self):
        for file in self.files:
            if 'prime' in file:
                self.prime_files.append(file)
            else:
                self.annot_files.append(file)

    def process_annot(self):
        if not os.listdir(self.gt_path):
            for file in self.prime_files:
                img1 = imageio.imread(file)
                img2 = imageio.imread(MESSIDORDataset.prime_to_annot(file))
                [x, y] = MESSIDORDataset.img_diff(img1, img2).max(-1).nonzero()
                self.annots.append([int(np.median(x)), int(np.median(y))])   
                print(f"done with {file}", end='\r')
            self.annots = np.asarray(self.annots, dtype=np.int64) 
            self.write_annots()
        else:
            self.load_annots()
            
    def write_annots(self):
        with open(os.path.join(self.gt_path, "annots.pkl"), 'wb') as f:
            pickle.dump({'prime_files': self.prime_files, 'annots': self.annots}, f)

    def load_annots(self):
        with open(os.path.join(self.gt_path, "annots.pkl"), 'rb') as f:
            data = pickle.load(f)
        assert set(self.prime_files) == set(data['prime_files']), 'files do not match.'
        self.annots = data['annots']

    def create_patch_mask(self):
        patch_rad = int(self.patch_diam/2)
        self.patch_mask = np.asarray([[1  if (x-patch_rad)**2 + (y-patch_rad)**2 < patch_rad ** 2 else 0 \
         for x in range(patch_rad*2)] for y in range(patch_rad*2)])
        self.patch_mask = np.expand_dims(self.patch_mask, -1)

    def valid_distances(self):
        self.valid_distance = int((self.img_diam - self.patch_diam)/2)
        self.img_center = int(self.img_diam/2)
        self.valid_pixels = [[x, y] for x in range(self.img_diam) for y in range(self.img_diam) if self.is_valid(x, y)]
        
    def crop_image(self, image):
        image = cv2.resize(image, dsize=self.img_size)
        return image[self.x_range[0]:self.x_range[1], self.y_range[0]:self.y_range[1], :]

    def shift_annot(self, annot):
        return [annot[0]-self.x_range[0], annot[1] - self.y_range[0]]

    def load_data(self, idx):
        path = self.prime_files[idx]
        image = self.crop_image(imageio.imread(path))
        annot = self.shift_annot(self.annots[idx])
        patches, labels = self.get_patches(image, annot)
        return {'path': path, 'img': image, 'annot': annot, 'patches': patches, 'labels': labels}

    def load_labels(self):
        self.labels = utils.load_csv(self.csv)
        self.cols = list(self.labels.columns)

    def __getitem__(self, idx):
        self.curr_name = Path(self.prime_files[idx]).stem
        print(f"done with {self.curr_name}", end='\r')
        return self.load_data(idx)

    def get_centers(self):
        return [random.choice(self.valid_pixels) for _ in range(self.n_patch)]
    
    def get_patches(self, image, annot):
        centers = self.get_centers()
        patches, labels = [], []
        patch_rad = int(self.patch_diam/2)

        for cx, cy in centers:
            patch = image[cx-patch_rad:cx+patch_rad, cy-patch_rad:cy+patch_rad, :]  * self.patch_mask

            patches.append(np.asarray(patch, dtype=np.uint8))
            labels.append(self.get_label(cx, cy, annot))

        return patches, labels
        
    def get_label(self, cx, cy, annot):
        vec = np.array([annot[0] - cx, annot[1] - cy], dtype=np.float64)
        return vec / norm(vec)

    def is_valid(self, x, y):
        return (x-self.img_center)**2 + (y-self.img_center)**2 < self.valid_distance**2

    @staticmethod
    def prime_to_annot(x, n=1):
        return x.replace('prime', f"-{n}")

def create_dataset():
    train_dataset = MESSIDORDataset(mode='train', img_type='tif', \
         dataroot=r"/mnt/c/Users/ssohr/OneDrive/Documents/eye-project/dataset/MESSIDOR/MESSIDOR")
    return data_utils.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)


if __name__ == "__main__":
    dataroot = r"/mnt/c/Users/ssohr/OneDrive/Documents/eye-project/dataset/MESSIDOR/MESSIDOR"
    ds = MESSIDORDataset(mode='train', img_type='tif', dataroot=dataroot)
    for i, item in enumerate(ds):
        pass