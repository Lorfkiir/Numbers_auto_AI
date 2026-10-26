import os
from PIL import Image

LABELS_LOAD_FOLDER = 'C:/Папки/GitHub/Numbers_auto_AI/data/train/new_labels'
IMAGES_LOAD_FOLDER = 'C:/Папки/GitHub/Numbers_auto_AI/data/train/images'
IMAGES_SAVE_FOLDER = 'C:/Папки/GitHub/Numbers_auto_AI/data/train/new_images'

class IMAGE:
    def __init__(self, load_path, save_path, x1, y1, x2, y2, img_width, img_height):
        self.load_path = load_path
        self.save_path = save_path
        self.x1 = round(x1 * img_width)
        self.y1 = round(y1 * img_height)
        self.x2 = round(x2 * img_width)
        self.y2 = round(y2 * img_height)

    def label(self):
        with Image.open(self.load_path) as img:
            box = (self.x1, self.y1, self.x2, self.y2)
            return img.crop(box)

    def save_img(self):
        img = self.label()
        img = img.convert('RGB')
        img.save(self.save_path, format='JPEG')
        img.close()

os.makedirs(LABELS_LOAD_FOLDER, exist_ok=True)

labels = sorted(
    f for f in os.listdir(LABELS_LOAD_FOLDER)
    if f.endswith('.txt')
)

images = sorted(
    f for f in os.listdir(IMAGES_LOAD_FOLDER)
    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
)

for label_file, image_file in zip(labels, images):

    image_path = os.path.join(IMAGES_LOAD_FOLDER, image_file)

    with Image.open(image_path) as img:
        img_width, img_height = img.size

    label_path = os.path.join(LABELS_LOAD_FOLDER, label_file)

    with open(label_path, 'r', encoding='utf-8') as file:
        for index, line in enumerate(file):
            parts = line.split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            x1, y1, x2, y2 = map(float, parts[1:])

            save_path = os.path.join(
                IMAGES_SAVE_FOLDER,
                f'{os.path.splitext(image_file)[0]}_{index}.jpg'
            )

            image = IMAGE(
                image_path, save_path,
                x1, y1, x2, y2,
                img_width, img_height
            )

            image.save_img()




