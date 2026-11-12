"""Архитектуры OCR (поиск номера) и REC (распознавание символов)."""
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.conv_1 = nn.Conv2d(3, 32, (3, 3), stride=2, padding=1)
        self.conv_2 = nn.Conv2d(32, 64, (3, 3), stride=2, padding=1)
        self.conv_3 = nn.Conv2d(64, 128, (3, 3), stride=2, padding=1)
        self.conv_4 = nn.Conv2d(128, 192, (3, 3), stride=2, padding=1)
        self.conv_5 = nn.Conv2d(192, 256, (3, 3), stride=2, padding=1)

        self.flatten = nn.Flatten()
        self.bn_1 = nn.BatchNorm2d(32)
        self.bn_2 = nn.BatchNorm2d(64)
        self.bn_3 = nn.BatchNorm2d(128)
        self.bn_4 = nn.BatchNorm2d(192)
        self.bn_5 = nn.BatchNorm2d(256)
        self.act = nn.ReLU()

        self.layer_1 = nn.Linear(256 * 20 * 20, 512)
        self.layer_2 = nn.Linear(512, 128)
        self.layer_3 = nn.Linear(128, out_channels)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.act(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.act(x)
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.act(x)
        x = self.conv_4(x)
        x = self.bn_4(x)
        x = self.act(x)
        x = self.conv_5(x)
        x = self.bn_5(x)
        x = self.act(x)

        x = self.flatten(x)

        x = self.layer_1(x)
        x = self.act(x)
        x = self.layer_2(x)
        x = self.act(x)
        out = self.layer_3(x)

        return out


class Rec_Model(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.conv_1 = nn.Conv2d(3, 32, (3, 3))
        self.conv_2 = nn.Conv2d(32, 64, (3, 3))
        self.conv_3 = nn.Conv2d(64, 128, (3, 3))
        self.conv_4 = nn.Conv2d(128, 256, (3, 3))

        self.bn_1 = nn.BatchNorm2d(32)
        self.bn_2 = nn.BatchNorm2d(64)
        self.bn_3 = nn.BatchNorm2d(128)
        self.bn_4 = nn.BatchNorm2d(256)

        self.act = nn.ReLU()

        self.flatten = nn.Flatten()
        self.max_pool = nn.MaxPool2d((2, 2))
        self.avg_pool = nn.AdaptiveAvgPool2d((4, 16))

        self.layer_1 = nn.Linear(256*4*16, 1024)
        self.layer_2 = nn.Linear(1024, 512)
        self.layer_3 = nn.Linear(512, out_channels)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Первая свертка
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.act(x)

        # Вторая свертка
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.act(x)

        # Уменьшение размера признаков
        x = self.max_pool(x)

        # Третья свертка
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.act(x)

        # Четвертая свертка
        x = self.conv_4(x)
        x = self.bn_4(x)
        x = self.act(x)

        # Приводим размер к фиксированному

        x = self.avg_pool(x)

        # Flatten
        x = self.flatten(x)

        # Linear
        x = self.layer_1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.layer_2(x)
        x = self.act(x)
        x = self.dropout(x)
        out = self.layer_3(x)
        out = out.view(-1, 9, 22)
        return out


OCR_str_info = '''
class Model(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.conv_1 = nn.Conv2d(3, 32, (3, 3), stride=2, padding=1)
        self.conv_2 = nn.Conv2d(32, 64, (3, 3), stride=2, padding=1)
        self.conv_3 = nn.Conv2d(64, 128, (3, 3), stride=2, padding=1)
        self.conv_4 = nn.Conv2d(128, 192, (3, 3), stride=2, padding=1)
        self.conv_5 = nn.Conv2d(192, 256, (3, 3), stride=2, padding=1)

        self.flatten = nn.Flatten()
        self.bn_1 = nn.BatchNorm2d(32)
        self.bn_2 = nn.BatchNorm2d(64)
        self.bn_3 = nn.BatchNorm2d(128)
        self.bn_4 = nn.BatchNorm2d(192)
        self.bn_5 = nn.BatchNorm2d(256)
        self.act = nn.ReLU()

        self.layer_1 = nn.Linear(256*20*20, 512)
        self.layer_2 = nn.Linear(512, 128)
        self.layer_3 = nn.Linear(128, out_channels)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.act(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.act(x)
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.act(x)
        x = self.conv_4(x)
        x = self.bn_4(x)
        x = self.act(x)
        x = self.conv_5(x)
        x = self.bn_5(x)
        x = self.act(x)

        x = self.flatten(x)

        x = self.layer_1(x)
        x = self.act(x)
        x = self.layer_2(x)
        x = self.act(x)
        out = self.layer_3(x)

        return out
'''

REC_str_info = '''

'''
