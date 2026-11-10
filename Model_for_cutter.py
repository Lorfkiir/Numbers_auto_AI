import torch

from Dataloader import PATH_TO_SAVE_MODEL
from Model import Model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

Model_for_cutter = Model(8).to(device)

Model_for_cutter.load_state_dict(torch.load(f'{PATH_TO_SAVE_MODEL}\\OCR_Detection_NN_.pt'))

Model_for_cutter.eval()
Model_for_cutter.to(device)
