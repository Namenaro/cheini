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


def visualise_foveas(foveas, name):
    foveas01 = utils.get_dataset(foveas)
    n = len(foveas01)
    plt.figure(figsize=(10, 2))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(foveas01[i], cmap='gray', vmax=1.0, vmin=0.0)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    name_for_pic = name+"foveas_.png"
    plt.savefig(name_for_pic)
    return name_for_pic


def dataset_to_pdf(a_dir):
    old_dir = os.getcwd()
    os.chdir(os.path.join(os.getcwd(), a_dir))
    story = []
    names = os.listdir(a_dir)
    areas = [os.path.join(a_dir, name, 'area.png.png') for name in names
             if os.path.isdir(os.path.join(a_dir, name))]
    foveas = [os.path.join(a_dir, name, 'foveas.pkl') for name in names
              if os.path.isdir(os.path.join(a_dir, name))]
    for i in range(len(areas)):
        name = names[i]
        ptext = '<font size=12>%s</font>' % name + "-----data-----"
        styles = getSampleStyleSheet()
        story.append(Paragraph(ptext, styles["Normal"]))

        name_for_pic = visualise_foveas(foveas[i], name)
        full_img_name = os.path.join(os.getcwd(), name_for_pic)
        im = utils.get_image_for_report(full_img_name, width=8 * cm)
        story.append(im)

        story.append(utils.get_image_for_report(areas[i], width=8 * cm))

    doc = SimpleDocTemplate("dataset-report.pdf", pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    doc.build(story)
    os.chdir(old_dir)


dataset_path = diropenbox(msg='select datset folder (a-la 5x5)')
dataset_to_pdf(dataset_path)

