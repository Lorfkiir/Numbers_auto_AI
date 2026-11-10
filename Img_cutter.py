import os
import torch

from PIL import Image
from torchvision import transforms
from Model_for_cutter import Model_for_cutter

device = 'cuda' if torch.cuda.is_available() else 'cpu'

PATH_TO_LOAD_IMAGES = r'C:\Папки\GitHub\Numbers_auto_AI\data\testph'
PATH_TO_SAVE_IMAGES = r'C:\Папки\GitHub\Numbers_auto_AI\data\testph1'

DIR = os.listdir(PATH_TO_LOAD_IMAGES)

transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
])

for files in DIR:
    img = Image.open(f"{PATH_TO_LOAD_IMAGES}\\{files}").convert('RGB')
    img = img.resize((640, 640))
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.inference_mode():
        prediction = Model_for_cutter(tensor)

    coords = prediction[0].cpu().tolist()
    re_coords = coords[:2] + coords[4:6]
    img = img.crop(re_coords)
    img.save(f"{PATH_TO_SAVE_IMAGES}\\{files}")
