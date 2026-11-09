import torch
import os
import json

from torchvision.transforms import v2
from PIL import Image

PATH_TO_TRAIN_DATASET = r'C:\Папки\GitHub\Numbers_auto_AI\data\train'
PATH_TO_VALID_DATASET = r'C:\Папки\GitHub\Numbers_auto_AI\data\val'
PATH_TO_SAVE_MODEL = r'C:\Папки\GitHub\Numbers_auto_AI\model'

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

        img = Image.open(path_img)
        coord = torch.tensor(self.dict_coords[name_file], dtype=torch.float)

        if self.transform is not None:
            img = self.transform(img)

        return img, coord

transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
    ]
)

train_dataset = Car_plate_OCR_Dataset(PATH_TO_TRAIN_DATASET, transform=transform)
valid_dataset = Car_plate_OCR_Dataset(PATH_TO_VALID_DATASET, transform=transform)
test_dataset = Car_plate_OCR_Dataset(PATH_TO_VALID_DATASET, transform=transform)
