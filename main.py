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


def get_fovea(image, x, y, hside):
    X1 = x - hside
    X2 = x + hside + 1
    Y1 = y - hside
    Y2 = y + hside + 1
    return image[Y1:Y2, X1:X2]


class CoordinateStore:
    def __init__(self, img_copy):
        self.points = []
        self.img_copy = img_copy

    def select_point(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(self.img_copy, (x, y), 0, (255, 0, 0), -1)
            self.points.append((x, y))
            print('x=' + str(x) + ' y=' + str(y))


class DatasetMaker:
    def __init__(self):
        self.hsides = [2, 3]
        self.folders = []
        for side in self.hsides:
            folder_name = str(side*2+1) +"x" + str(side*2+1)
            self.folders.append(folder_name)
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
        self.root_folder = os.getcwd()
        self.show_saccada_to_user = True

    def make_foveas_seq(self, points, img, hside):
        foveas = []
        for point in points:
            fovea_img = get_fovea(image=img, x=point[0], y=point[1], hside=hside)
            foveas.append(fovea_img)
        utils.save_object(points, 'points')
        utils.save_object(foveas, 'foveas')
        for i in range(len(points)):
            utils.save_img_scaled(str(i), foveas[i], scaling_factor=5)
        self.visualise_saccade(foveas)

    def get_area_of_saccada(self, points, img_copy, margin):
        xs = np.array(points)[:, 0]
        ys = np.array(points)[:, 1]
        x_min = min(xs) - margin
        x_max = max(xs)
        y_min = min(ys) - margin
        y_max = max(ys)
        return img_copy[y_min:y_max, x_min:x_max]


    def visualise_saccade(self, foveas):
        if self.show_saccada_to_user is False:
            return
        foveas01 = utils.scale_dataset_to01(foveas)
        foveas01 = np.array(foveas01)
        n = len(foveas01)
        plt.figure(figsize=(10, 2))
        for i in range(n):
            # display original
            ax = plt.subplot(2, n, i + 1)
            ax.set_title("original " + str(i))
            plt.imshow(foveas01[i], cmap='gray', vmax=1.0, vmin=0.0)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.savefig("saccada.png")
        plt.show()

    def make_saccada(self, saccade_name, path_to_pic):
        img = cv2.imread(path_to_pic, cv2.IMREAD_GRAYSCALE)  # not draw on it!!!
        img_copy = copy.deepcopy(img)  # draw on it
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2RGB)
        cv2.imshow('picture', img_copy)
        coordinateStore = CoordinateStore(img_copy=img_copy)
        cv2.setMouseCallback('picture', coordinateStore.select_point)

        while(1):
            cv2.imshow('picture', img_copy)
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
        cv2.destroyAllWindows()

        # сохранение результатов:
        for k in range(len(self.hsides)):
            hside = self.hsides[k]
            folder_sub_seria = os.path.join(self.root_folder, self.folders[k])
            os.chdir(folder_sub_seria)
            utils.setup_folder_for_results(saccade_name)
            self.make_foveas_seq(points=coordinateStore.points, img=img, hside=hside)
            area_saccada = self.get_area_of_saccada(points=coordinateStore.points, img_copy=img_copy, margin=hside*3)
            utils.save_img_scaled("area", area_saccada, scaling_factor=3)


    def main_cycle(self):
        num = 0
        while (1):
            path_to_pic = easygui.fileopenbox(msg='выбрать картинку', filetypes=[["*.png", "*.jpg", "*.jpeg", 'картинки']])
            if path_to_pic is None:
                return
            saccade_name = str(num)  # utils.ask_user_for_name()
            self.make_saccada(saccade_name=saccade_name, path_to_pic=path_to_pic)
            num += 1
            os.chdir(self.root_folder)  # обратно в папку серии


utils.setup_folder_for_results(os.path.join(os.getcwd(),"results2"))
dataset_maker = DatasetMaker()
dataset_maker.main_cycle()
