import torch
import torchvision
import os
import numpy as np
import json
import matplotlib.pyplot as plt

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

PATH_TO_TRAIN = r'C:\Папки\GitHub\Numbers_auto_AI\data\train'
PATH_TO_VALID = r'C:\Папки\GitHub\Numbers_auto_AI\data\val'
PATH_TO_TEST = r'C:\Папки\GitHub\Numbers_auto_AI\data\test'

class Car_plate_OCR_Dataset():
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform

        self.list_name_file = os.listdir(self.path)
        if 'coords.json' in self.list_name_file:
            self.list_name_file.remove('coords.json')

        self.len_dataset = len(self.list_name_file)

        with open(os.path.join(self.path, 'coords.json'), 'r') as f:
            self.dict_coords = json.load(f)

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, index):
        name_file = self.list_name_file[index]
        path_img = os.path.join(self.path, name_file)

        img = np.array(Image.open(path_img))
        coord = np.array(self.dict_coords[name_file])

        if self.transform is not None:
            img = self.transform(img)

        return img, coord

train_dataset = Car_plate_OCR_Dataset(PATH_TO_TEST)
valid_dataset = Car_plate_OCR_Dataset(PATH_TO_VALID)
test_dataset = Car_plate_OCR_Dataset(PATH_TO_TEST)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

