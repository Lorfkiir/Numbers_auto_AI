import torch.nn as nn

class Model(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.conv_1 = nn.Conv2d(3, 32, (3, 3), 2)
        self.conv_2 = nn.Conv2d(32, 64, (3, 3), 2)
        self.conv_3 = nn.Conv2d(64, 128, (3, 3), 2)
        self.conv_4 = nn.Conv2d(128, 256, (3, 3), 2)

        self.flatten = nn.Flatten()

        self.act = nn.ReLU()

        self.layer_1 = nn.Linear(256*39*39, 128)
        self.layer_2 = nn.Linear(128, 64)
        self.layer_3 = nn.Linear(64, out_channels)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.act(x)
        x =self.conv_2(x)
        x = self.act(x)
        x = self.conv_3(x)
        x = self.act(x)
        x = self.conv_4(x)
        x = self.act(x)

        x = self.flatten(x)

        x = self.layer_1(x)
        x = self.act(x)
        x = self.layer_2(x)
        x = self.act(x)
        x = self.dropout(x)
        out = self.layer_3(x)

        return out

str_info = '''
class Model(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.conv_1 = nn.Conv2d(3, 32, (3, 3), 2)
        self.conv_2 = nn.Conv2d(32, 64, (3, 3), 2)
        self.conv_3 = nn.Conv2d(64, 128, (3, 3), 2)
        self.conv_4 = nn.Conv2d(128, 256, (3, 3), 2)

        self.flatten = nn.Flatten()

        self.act = nn.ReLU()

        self.layer_1 = nn.Linear(256*39*39, 128)
        self.layer_2 = nn.Linear(128, 64)
        self.layer_3 = nn.Linear(64, out_channels)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.act(x)
        x =self.conv_2(x)
        x = self.act(x)
        x = self.conv_3(x)
        x = self.act(x)
        x = self.conv_4(x)
        x = self.act(x)

        x = self.flatten(x)

        x = self.layer_1(x)
        x = self.act(x)
        x = self.layer_2(x)
        x = self.act(x)
        x = self.dropout(x)
        out = self.layer_3(x)

        return out
'''