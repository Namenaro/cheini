# -*- coding: utf-8 -*
print ("MAIN_FIT")
############################ STAGE 1 #############################################
###################### Prerocess data ############################################
##################################################################################
import easygui
import utils
import cv2
import numpy as np

# load from pkl
points_pkl = easygui.fileopenbox(msg='выбрать файл points.pkl (коордитаы фиксаций)', filetypes=["*.pkl"])
if points_pkl is None:
    exit()
points = utils.open_file(points_pkl)

foveas255_pkl = easygui.fileopenbox(msg='выбрать файл foveas.pkl', filetypes=["*.pkl"])
if foveas255_pkl is None:
    exit()
foveas255 = utils.open_file(foveas255_pkl)

# norm [0, 255] to [0, 1]
foveas01 = utils.scale_dataset_to01(foveas255)
foveas01 = np.array(foveas01)
print ('input data shape: ' + str(foveas01.shape))

############################ STAGE 2 #############################################
########################## Create net ############################################
##################################################################################


# create
# fit

############################ STAGE 3 #############################################
########################## evaluate res ##########################################
##################################################################################

# visualise reconstruction
# visualise manifold
# weights / biases  распределения
# energy of temporal changes
# energy (entropy) per frame


############################ STAGE 4 #############################################
########################## save summary ##########################################
##################################################################################

# start dialog what to save


