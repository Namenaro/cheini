# -*- coding: utf-8 -*
import easygui
import copy
import cv2
import numpy as np
import utils
import matplotlib.pyplot as plt
import os


events = [i for i in dir(cv2) if 'EVENT' in i]
print (events)

HSIDE = 2



def make_seq(name_seq):
    path = easygui.fileopenbox(msg='выбрать картинку', filetypes=[["*.png", "*.jpg","*.jpeg", 'картинки']])
    if path is None:
        return
    print("selected" + path)
    name = str(name_seq) #utils.ask_user_for_name()
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
        def __init__(self):
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


    coordinateStore1 = CoordinateStore()
    cv2.setMouseCallback(name, coordinateStore1.select_point)

    while(1):
        cv2.imshow(name, img_copy)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()

    # сохранение результатов:
    utils.setup_folder_for_results(name)
    utils.save_object(coordinateStore1.points, 'points')
    utils.save_object(coordinateStore1.foveas, 'foveas')
    utils.save_object(path, 'name_of_image')
    for i in range(len(coordinateStore1.points)):
        utils.save_img_scaled(str(i),coordinateStore1.foveas[i], scaling_factor=5)
    restored_points = utils.open_file('points.pkl')
    print("points: " + str(restored_points))

    # visualise foveas
    foveas01 = utils.scale_dataset_to01(coordinateStore1.foveas)
    foveas01 = np.array(foveas01)
    n = len(foveas01)
    plt.figure(figsize=(10, 2))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        ax.set_title( "original " + str(i))
        plt.imshow(foveas01[i], cmap='gray', vmax=1.0, vmin=0.0)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(name + "_.png")
    plt.show()


num = 0
folder_full_path = os.getcwd()
while(1):
    make_seq(num)
    num += 1
    os.chdir(folder_full_path)  # обратно в папку серии
