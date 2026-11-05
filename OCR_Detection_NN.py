import torch
import torch.nn as nn
import torchvision
import os
import numpy as np
import json
import matplotlib.pyplot as plt

from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from tqdm import tqdm

plt.style.use('dark_background')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 50
train_loss = []
train_acc = []
val_loss = []
val_acc = []

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

        img = Image.open(path_img)
        coord = torch.tensor(self.dict_coords[name_file], dtype=torch.float)

        if self.transform is not None:
            img = self.transform(img)

        return img, coord

class Model(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.layer_1 = nn.Linear(input, 128)
        self.layer_2 = nn.Linear(128, output)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.layer_1(x)
        x = self.act(x)
        out = self.layer_2(x)
        return out


# Трансформация для передачи правильной разметки 640х640х3(RGB)

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

train_dataset = Car_plate_OCR_Dataset(PATH_TO_TRAIN, transform=transform)
valid_dataset = Car_plate_OCR_Dataset(PATH_TO_VALID, transform=transform)
test_dataset = Car_plate_OCR_Dataset(PATH_TO_TEST, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Объявляем и указываем параметры нейронной сети

OCR_Detection_NN = Model(3*640*640, 2).to(device)

loss_regression = nn.MSELoss()
opt_regression = torch.optim.Adam(OCR_Detection_NN.parameters(), lr = 0.001)


# Цикл обучения нейронной сети

for epoch in range(EPOCHS):

    # Тренировка модели
    OCR_Detection_NN.train()
    running_train_loss = []
    true_answers = 0
    train_loop = tqdm(train_loader, leave=False)
    for x, targets in train_loop:
        # Данные
        # (batch_size, 3, 640, 640) -> (batch_size, 3*640*640)
        x = x.reshape(x.shape[0], -1).to(device)
        targets = targets.to(device)

        # Прямой проход и расчет ошибки модели
        prediction = OCR_Detection_NN(x)
        loss = loss_regression(prediction, targets)

        # Обратный проход
        opt_regression.zero_grad()
        loss.backward()
        # Шаг оптимизации
        opt_regression.step()

        running_train_loss.append(loss.item())
        mean_train_loss = sum(running_train_loss) / len(running_train_loss)

        true_answers += (torch.round(prediction) == targets).all(dim=1).sum().item()

        train_loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}], train loss: {mean_train_loss:.4f}")

    # Расчет значения метрики
    running_train_acc = true_answers / len(train_dataset)

    # Сохранение значения функции потерь и метрики
    train_loss.append(mean_train_loss)
    train_acc.append(running_train_acc)

    # Validation
    OCR_Detection_NN.eval()
    with torch.no_grad():
        running_val_loss = []
        true_answers = 0
        for x, targets in valid_loader:
            # Данные
            # (batch_size, 3, 640, 640) -> (batch_size, 3*640*640)
            x = x.reshape(x.shape[0], -1).to(device)
            targets = targets.to(device)

            # Прямой проход и расчет ошибки модели
            prediction = OCR_Detection_NN(x)
            loss = loss_regression(prediction, targets)

            running_val_loss.append(loss.item())
            mean_val_loss = sum(running_val_loss) / len(running_val_loss)

            true_answers += (torch.round(prediction) == targets).all(dim=1).sum().item()

        # Расчет значения метрики
        running_val_acc = true_answers / len(valid_dataset)

        # Сохранение значения функции потерь и метрики
        val_loss.append(mean_val_loss)
        val_acc.append(running_val_acc)

        print(f"Epoch [{epoch+1}/{EPOCHS}], train_loss = {mean_train_loss:.4f}, train_acc = {running_train_acc:.4f}, val_loss = {mean_val_loss:.4f}, val_acc = {running_val_acc:.4f}")

plt.plot(train_loss)
plt.plot(val_loss)
plt.legend(['loss_train', 'loss_val'])
plt.show()

plt.plot(train_acc)
plt.plot(val_acc)
plt.legend(['acc_train', 'acc_val'])
plt.show()







