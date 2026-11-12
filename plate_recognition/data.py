"""Датасеты и DataLoader для задач OCR и REC."""
import json
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from .config import (
    PATH_TO_TRAIN_DATASET,
    PATH_TO_VALID_DATASET,
    PATH_TO_TEST_DATASET,
    PATH_TO_TRAIN_DATASET1,
    PATH_TO_VALID_DATASET1,
    PATH_TO_TEST_DATASET1,
    PATH_TO_SAVE_MODEL,
)
from .decoder import encode

class Car_plate_OCR_Dataset():
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform

        self.list_name_file = sorted(os.listdir(self.path))
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


class Car_plate_REC_Dataset():
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform

        self.list_name_file = sorted(os.listdir(self.path))

        if 'data.json' in self.list_name_file:
            self.list_name_file.remove('data.json')

        self.len_dataset = len(self.list_name_file)

        with open(os.path.join(self.path, 'data.json'), 'r', encoding='utf-8') as f:
            self.dict_data = json.load(f)

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, index):
        name_file = self.list_name_file[index]
        path_img = os.path.join(self.path, name_file)

        img = Image.open(path_img).convert('RGB')

        text = self.dict_data[name_file]
        data = torch.tensor(encode(text), dtype=torch.float32)

        if self.transform is not None:
            img = self.transform(img)

        return img, data


OCR_transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
    ]
)

REC_transform = v2.Compose(
    [
        v2.ToImage(),
        v2.Resize((48, 160)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
    ]
)


OCR_train_dataset = Car_plate_OCR_Dataset(PATH_TO_TRAIN_DATASET, transform=OCR_transform)
OCR_valid_dataset = Car_plate_OCR_Dataset(PATH_TO_VALID_DATASET, transform=OCR_transform)
OCR_test_dataset = Car_plate_OCR_Dataset(PATH_TO_TEST_DATASET, transform=OCR_transform)
OCR_train_loader = DataLoader(OCR_train_dataset, batch_size=4, shuffle=True)
OCR_valid_loader = DataLoader(OCR_valid_dataset, batch_size=4, shuffle=False)
OCR_test_loader = DataLoader(OCR_test_dataset, batch_size=4, shuffle=False)

REC_train_dataset = Car_plate_REC_Dataset(PATH_TO_TRAIN_DATASET1, transform=REC_transform)
REC_valid_dataset = Car_plate_REC_Dataset(PATH_TO_VALID_DATASET1, transform=REC_transform)
REC_test_dataset = Car_plate_REC_Dataset(PATH_TO_TEST_DATASET1, transform=REC_transform)
REC_train_loader = DataLoader(REC_train_dataset, batch_size=4, shuffle=True)
REC_valid_loader = DataLoader(REC_valid_dataset, batch_size=4, shuffle=False)
REC_test_loader = DataLoader(REC_test_dataset, batch_size=4, shuffle=False)
