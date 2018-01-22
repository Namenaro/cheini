from easygui import multenterbox
import _pickle as pickle
import cv2
import os
import glob

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
    with open(filename + ".pkl", 'rb') as input:
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

def resize_img(img, scaling_factor ):
    matrix1 = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_NEAREST)
    cv2.imshow("INTER_NEAREST", matrix1)