import os

LOAD_PATH = "C:/Папки/GitHub/Numbers_auto_AI/data/test"
SAVE_FOLDER = "C:/Папки/GitHub/Numbers_auto_AI/data/new_test"

class Image():

    def __init__(self,label_path, new_label_path):
        self.label_path = label_path
        self.new_label_path = new_label_path

    def save_image_with_frame_layout(self, text):
        with open(self.label_path, "r") as file:
            with open(self.new_label_path, "w") as new_file:
                for elements in file:
                    image = Coordinates_of_Number(*elements.split())
                    coordinates = image.frame_layout()
                    new_file.write(" ".join(map(str, coordinates)) + "\n")


class Coordinates_of_Number():

    def __init__(self, type, x_center, y_center, width, height):

        self.type = int(type)
        self.x_center = float(x_center)
        self.y_center = float(y_center)
        self.width = float(width)
        self.height = float(height)

    def frame_layout(self):

        type = int(self.type)
        x1 = float(self.x_center - self.width/2)
        y1 = float(self.y_center - self.height/2)
        x2 = float(self.x_center + self.width/2)
        y2 = float(self.y_center + self.height/2)

        return (type, x1, y1, x2, y2)

for file in os.listdir(LOAD_PATH):
    label_path = os.path.join(LOAD_PATH, file)
    new_label_path = os.path.join(SAVE_FOLDER, file)

    image = Image(label_path, new_label_path)
    image.save_image_with_frame_layout("")



