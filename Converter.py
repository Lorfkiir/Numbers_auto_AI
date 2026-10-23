LOAD_PATH = "ПУТЬ К ДАТАСЕТУ 1"
SAVE_FOLDER = "ПУТЬ К НОВОМУ ДАТАСЕТУ"

class Image():
    '''РЕАЛИЗОВАТЬ ЗДЕСЬ ИМПОРТ .txt ИЗ ДАННЫХ К КАЖДОМУ ФОТО'''
    pass

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

    def save_image_with_frame_layout(self):
        '''ДОПИСАТЬ СОХРАНЕНИЕ ИЗОБРАЖЕНИЯ С X1,X2,Y1,Y2'''
        pass


Det1 = Coordinates_of_Number(x_center = 100, y_center = 50, width = 10, height = 30)
print(Det1.frame_layout())




