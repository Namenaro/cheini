# -*- coding: utf-8 -*
from easygui import multenterbox
import _pickle as pickle
import matplotlib.pyplot as plt
import os
import glob
import easygui
import cv2
import numpy as np
from reportlab.lib import utils
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image

def ask_user_for_name():
    msg = "Enter name for folder"
    title = "to save results"
    fieldNames = ["Name"]
    fieldValues = []  # we start with blanks for the values
    fieldValues = multenterbox(msg,title, fieldNames)

    # make sure that none of the fields was left blank
    while 1:
        if fieldValues is None: break
        errmsg = ""
        for i in range(len(fieldNames)):
            if fieldValues[i].strip() == "":
                errmsg += ('"%s" is a required field.\n\n' % fieldNames[i])
        if errmsg == "":
            break # no problems found
        fieldValues = multenterbox(errmsg, title, fieldNames, fieldValues)

    print("Reply was: %s" % str(fieldValues))
    return fieldValues[0]


def save_object(obj, filename):
    pickle.dump(obj, open(filename + ".pkl", "wb"))

def open_file(filename):
    obj = None
    with open(filename , 'rb') as input:
        obj = pickle.load(input)
    return obj

def save_img_scaled(png_name_without_extension, img, scaling_factor):
    img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_NEAREST)
    full_name = png_name_without_extension + ".png"
    cv2.imwrite(full_name + ".png", img)
    return full_name

def setup_folder_for_results(main_folder='results'):
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)
    else:
        files = glob.glob('/' + main_folder + '/*')
        for f in files:
            os.remove(f)
    os.chdir(main_folder)

def scale_dataset_to01(dataset):
    result = []
    for tensor in dataset:
        tensor01 = tensor/255.0
        result.append(tensor01)
    return result


# load from pkl
def get_dataset(foveas255_pkl=None):
    if foveas255_pkl is not None:
        pass
    else:
        foveas255_pkl = easygui.fileopenbox(msg='выбрать файл foveas.pkl', filetypes=["*.pkl"])
        if foveas255_pkl is None:
            exit()
        print("opened file with foveas: " + foveas255_pkl)
    foveas255 = open_file(foveas255_pkl)

    # norm [0, 255] to [0, 1]
    foveas01 = scale_dataset_to01(foveas255)
    foveas01 = np.array(foveas01)
    print ('input data shape: ' + str(foveas01.shape))
    return foveas01

def resize_img(img, scaling_factor ):
    matrix1 = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_NEAREST)
    cv2.imshow("INTER_NEAREST", matrix1)


def draw_mat(matrix):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    mmin = matrix.min()
    mmax = matrix.max()
    abs_max = max(abs(mmax), abs(mmin))
    cax = ax.matshow(matrix, vmin=-abs_max, vmax=abs_max)
    fig.colorbar(cax)
    plt.show()

def trash():
    a = np.array([[2.1, 3], [4, 5]])
    b = np.array([[3, 4], [5, 6]])
    abs_dif = np.absolute(b-a)
    energy = abs_dif.sum()
    print (str(energy))

def energy_change(a, b):
    if a.shape == b.shape:
        abs_dif = np.absolute(b - a)
        energy = abs_dif.sum()
        num_of_items = np.prod(a.shape[:])
        return energy/num_of_items
    else:
        return None

def get_image_for_report(path, width=8*cm):
    img = utils.ImageReader(path)
    iw, ih = img.getSize()
    aspect = ih / float(iw)
    return Image(path, width=width, height=(width * aspect))

# 9. Тест шумом
def add_noise_to_dataset(dataset, noise_factor=0.01, to_plt=True):
    x_train_noisy = dataset + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=dataset.shape)
    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    n = len(dataset)
    if to_plt:
        plt.figure(figsize=(10, 2))
        for i in range(n):
            # display original
            ax = plt.subplot(2, n, i + 1)
            ax.set_title("before " + str(i))
            plt.imshow(dataset[i], cmap='gray', vmax=1.0, vmin=0.0)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display with noise
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(x_train_noisy[i], cmap='gray', vmax=1.0, vmin=0.0)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    return x_train_noisy

def save_all(encoder, decoder, autoencoder):
    #https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
    encoder.save('encoder.h5')
    decoder.save('decoder.h5')
    autoencoder.save('autoencoder.h5')