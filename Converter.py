import os

LOAD_PATH = "ПУТЬ К ДАТАСЕТУ 1"
SAVE_FOLDER = "ПУТЬ К НОВОМУ ДАТАСЕТУ"

class Image():

    def __init__(self, image_path, label_path):
        self.image_path = image_path
        self.label_path = label_path

    def save_image_with_frame_layout(self, text):
        with open(self.label_path, "w") as file:
            file.write(text)


class Coordinates_of_Number():

    def __init__(self, x_center, y_center, width, height):

        self.x_center = x_center
        self.y_center = y_center
        self.width = width
        self.height = height

    def frame_layout(self):
        x1 = int(self.x_center - self.width/2)
        y1 = int(self.y_center - self.height/2)
        x2 = int(self.x_center + self.width/2)
        y2 = int(self.y_center + self.height/2)

        return (x1,y1,x2,y2)



