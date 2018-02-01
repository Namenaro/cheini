# -*- coding: utf-8 -*
import os
directory = 'C:\\Users\\neuro\\PycharmProjects\\cheini\\big dataset'
def get_dataset(a_dir):
    return [os.path.join(a_dir, name, 'foveas.pkl') for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

subfolders = get_dataset(directory)
print (subfolders)
