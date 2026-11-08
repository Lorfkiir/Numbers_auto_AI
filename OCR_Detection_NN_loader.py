import torch
import torch.nn as nn
import torchvision
import os
import numpy as np
import json
import matplotlib.pyplot as plt

from torchvision.transforms import v2
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from TRAIN_OCR_Detection_NN import OCR_Detection_NN


plt.style.use('dark_background')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

PATH_TO_LOAD_MODEL_PARAMETRS = r''
PATH_TO_DATASET = r''

MODEL_PARAMETRS = torch.load(PATH_TO_LOAD_MODEL_PARAMETRS)

Load_Model = OCR_Detection_NN(3*640*640, 8).to(device)
Load_Model.load_state_dict(MODEL_PARAMETRS)








