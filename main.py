import easygui
import copy
import cv2
import numpy as np
import utils


events = [i for i in dir(cv2) if 'EVENT' in i]
print (events)

HSIDE = 2

path = easygui.fileopenbox()
print("selected" + path)
name = utils.ask_user_for_name()
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # not draw on it!!!
img_copy = copy.deepcopy(img)  # draw on it
img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2RGB)
cv2.imshow(name, img_copy)

def get_fovea(image, x, y, hside):
    X1 = x - hside
    X2 = x + hside + 1
    Y1 = y - hside
    Y2 = y + hside + 1
    return image[Y1:Y2 ,X1:X2 ]

class CoordinateStore:
    def __init__(self, image):
        self.points = []
        self.foveas = []

    def select_point(self,event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDBLCLK:
                cv2.circle(img_copy,(x,y), 0,(255,0,0),-1)
                self.points.append((x,y))
                print('x=' + str(x)+' y=' + str(y))
                fovea_img = get_fovea(image=img, x=x, y=y, hside=HSIDE)
                self.foveas.append(fovea_img)
                cv2.imshow(name + 'fovea', fovea_img)


coordinateStore1 = CoordinateStore(image=img)
cv2.setMouseCallback(name, coordinateStore1.select_point)

while(1):
    cv2.imshow(name, img_copy)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()

# сохранение результатов:
utils.setup_folder_for_results(name)
utils.save_object(coordinateStore1, 'points and imgs')
for i in range(len(coordinateStore1.points)):
    utils.save_img_scaled(str(i),coordinateStore1.foveas[i], scaling_factor=5)
restored = utils.open_file('points and imgs')
print("points: " + str(restored.points))
