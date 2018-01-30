# -*- coding: utf-8 -*

import itertools
import one_experiment_report
import utils
import simple_nets
from math import floor, ceil
import matplotlib.pyplot as plt
import numpy as np
import os

from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras import optimizers
import time
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
import pandas as pd


# варьируем один (или несколько) гиперпараметр - проводим таким образом серию экспериментов,
# результаты серии сводим в единый отчет: таблица из 2 столбцов (что вариьровали) и (за чем следили)
#one_experiment_report.main()

class Serial:
    def __init__(self):
        self.code_len = None
        self.num_epochs = None
        self.activation = None

    def get_arrays(self):
        self.code_len = [2]
        self.num_epochs = [100, 200]
        self.activation = ['sigmoid']
        self.dataset = ['C:\\Users\\neuro\\PycharmProjects\\cheini\\5x5\\const_part_and_dyn_other\\foveas.pkl']

    def get_all_cominations(self):
        """
        :return: список словарей - всех возхможных комбинаций значений гиперпараметров
        """
        def enumdict(listed):
            myDict = {}
            for i, x in enumerate(listed):
                myDict[i] = x
            return myDict
        hypermapars_arrays = self.__dict__
        names = hypermapars_arrays.keys()
        enumerated_names = enumdict(names) # например {0: 'code_len', 1: 'activation', 2: 'num_epochs'}

        n_hyperparams = len(enumerated_names.keys())
        a = [None] * n_hyperparams
        for k in enumerated_names.keys():
            name = enumerated_names[k]
            a[k] = hypermapars_arrays[name]
        all_combinations = list(itertools.product(*a))
        all_dicts = []
        for combination in all_combinations:
            d = {}
            for i in enumerated_names.keys():
                name = enumerated_names[i]
                d[name] = combination[i]
            all_dicts.append(d)
        return all_dicts

    def make_experiments(self, all_dicts):
        summaries = []
        experiment_id = 0
        folder_name = utils.ask_user_for_name()  # выбрать имя серии
        if folder_name is None:
            exit()
        utils.setup_folder_for_results(folder_name)
        folder_full_path = os.getcwd()
        for params in all_dicts:
            utils.setup_folder_for_results(str(experiment_id))  # имя эксперимента в серии
            e = Experiment(params)
            summary = e.run_it()
            summary['experiment_name'] = experiment_id
            summaries.append(summary)
            experiment_id += 1
            os.chdir(folder_full_path)  # обратно в папку серии
        return summaries

class Experiment:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)

    def run_it(self):
        print("RUN: " + str(self.__dict__))
        # вытаскиваем датасет из файла
        foveas01 = utils.get_dataset(self.dataset)

        # создаем и обучаем модельку
        en, de, ae = simple_nets.create_ae_ZINA(encoding_dim=self.code_len,
                                                input_data_shape=foveas01[0].shape,
                                                a_koef_reg=0.001,
                                                koef_reg=0.0001,
                                                activation_on_code=self.activation,
                                                drop_in_decoder=0.1,
                                                drop_in_encoder=0.1)

        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        ae.compile(optimizer=sgd, loss='mean_squared_error')

        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
        history = ae.fit(foveas01, foveas01,
                         epochs=self.num_epochs,
                         batch_size=ceil(len(foveas01) / 2),
                         shuffle=True,
                         validation_data=(foveas01, foveas01),
                         callbacks=[early_stopping])

        # по результатам обучения на этом датасетке генерим репорт
        report = one_experiment_report.ReportOnPath(ae=ae, en=en, de=de,
                                                    dataset=foveas01,
                                                    history_obj=history)
        report.create_summary()
        summary = report.end()
        utils.save_all(encoder=en, decoder=de, autoencoder=ae)
        return summary

if __name__ == "__main__":
    utils.setup_folder_for_results("SERIES")
    s = Serial()
    s.get_arrays()
    n = len(s.get_all_cominations())
    print ("there will be :"  + str(n) + " experiments!")
    summaries = s.make_experiments(s.get_all_cominations())
    results_table = pd.DataFrame(summaries)
    print(results_table)



