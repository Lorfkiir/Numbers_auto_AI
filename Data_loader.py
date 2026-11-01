import torch
import torchvision
import os
import numpy as np
import json
import matplotlib.pyplot as plt

from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

PATH_TO_TRAIN = ''
PATH_TO_VALID = ''
PATH_TO_TEST = ''

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
        pass
    def __getitem__(self, index):
        pass
