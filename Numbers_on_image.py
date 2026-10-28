import numpy as np
import torch
from PIL import Image

class IMAGE():
    def __init__(self, path):
        self.path = path

    def tensor_from_image(self):
        img = Image.open(self.path).convert('RGB')
        img = img.resize((320, 320))

        array = np.array(img, dtype=np.float32) / 255.0

        tensor = torch.from_numpy(array).permute(2, 0, 1)
        tensor = tensor.unsqueeze(0)

        return tensor

# Написать модель -> реализовать загрузку tensor(а)
