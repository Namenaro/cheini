# -*- coding: utf-8 -*

import utils
import matplotlib.pyplot as plt
import numpy as np
import itertools
import os

import time
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from easygui import diropenbox

def dataset_to_truncated(a_dir):
    old_dir = os.getcwd()
    os.chdir(os.path.join(os.getcwd(), a_dir))
    names = os.listdir(a_dir)
    foveas_pkls = [os.path.join(os.getcwd(), name, 'foveas.pkl') for name in names
              if os.path.isdir(os.path.join(a_dir, name))]
    os.mkdir('super_truncated_version')
    folder_dataset = os.path.join(os.getcwd(), 'super_truncated_version')

    for i in range(len(names)):
        os.chdir(folder_dataset)
        name = names[i]
        foveas = utils.open_file(foveas_pkls[i])
        if len(foveas)>3:
            tr_foveas = []
            tr_foveas.append(foveas[0])
            tr_foveas.append(foveas[-1])
            os.mkdir(name)
            os.chdir(name)
            utils.save_object(tr_foveas, 'foveas')
            print(name)

    os.chdir(old_dir)





dataset_path = diropenbox(msg='select datset folder (a-la 5x5)')
dataset_to_truncated(dataset_path)